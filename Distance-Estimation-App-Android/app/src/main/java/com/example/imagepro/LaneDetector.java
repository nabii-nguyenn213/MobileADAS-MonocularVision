package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.opencv.core.MatOfPoint;

/** Port of the supplied Python lane preprocessing, RANSAC and temporal lane tracking.
 * Coordinates returned to CameraActivity are upright, original-frame pixels.
 * Only the lane subsystem is changed; YOLO and telemetry APIs remain unchanged.
 */
public final class LaneDetector implements AutoCloseable {
    private static final String TAG="LANE_PERF";
    private static final int BASE=640, SAMPLES=30, MIN_POINTS=5;
    private static final double ROI_TOP=.30, ROI_BOTTOM=.98, CENTER=.5;
    private static final double MIN_SPAN=.001, RESIDUAL=.008, MIN_WIDTH=.12, MAX_WIDTH=.90;
    private static final double ALPHA=.25, MAX_JUMP=.045, HOLD_SECONDS=.5;
    private static final int LANE_INTERVAL=1;
    private static final Scalar LEFT_COLOR=new Scalar(0,255,0,255);
    private static final Scalar RIGHT_COLOR=new Scalar(255,0,0,255);
    private final Interpreter interpreter;
    private final int inputH,inputW,outputH,outputW;
    private final boolean inputNchw,outputNchw;
    private final ByteBuffer input;
    private final byte[] rgbBytes;
    private final float[][][][] laneOutput;
    private final float[][][][] drivableOutput;
    private final Map<Integer,Object> outputs=new HashMap<>();
    private final Mat resized=new Mat(),padded=new Mat(),rgb=new Mat(),margin=new Mat(),cropped=new Mat(),fullMargin=new Mat();
    private final Mat sceneGray=new Mat(),thumbnail=new Mat();
    private final ArrayList<Point> displayLeft=new ArrayList<>(),displayRight=new ArrayList<>();
    private final ArrayList<Point> reliableLeft=new ArrayList<>(),reliableRight=new ArrayList<>();
    private final SideTrack leftTrack=new SideTrack(),rightTrack=new SideTrack();
    private boolean closed=false, virtualMode=false;
    private int frameCount=0,maskPixels,roiPixels,candidateCount,lineCount;
    private String failure="not_run";
    private long lastObservedNs=0;
    private int lastFrameW=-1,lastFrameH=-1;
    private int logFrames=0;
    private double prepSum=0,inferSum=0,postSum=0;

    private static final class Line {
        double a,b,low,high;
        Line(double a,double b,double low,double high){this.a=a;this.b=b;this.low=low;this.high=high;}
        double x(double y){return a*y+b;}
        double bottom(){return x(ROI_BOTTOM);}
        Line copy(){return new Line(a,b,low,high);}
    }
    private static final class SideTrack {
        Line current,target,lastObserved;
        long lastSeenNs;
        int hits;
        double confidence;
        boolean valid;
        void clear(){current=target=lastObserved=null;lastSeenNs=0;hits=0;confidence=0;valid=false;}
    }

    public LaneDetector(AssetManager assets,String modelName) throws IOException {
        Interpreter.Options options=new Interpreter.Options();
        options.setNumThreads(2);
        options.setUseXNNPACK(false);
        try(AssetFileDescriptor fd=assets.openFd(modelName);
            FileInputStream stream=new FileInputStream(fd.getFileDescriptor());
            FileChannel channel=stream.getChannel()) {
            interpreter=new Interpreter(channel.map(FileChannel.MapMode.READ_ONLY,fd.getStartOffset(),fd.getDeclaredLength()),options);
        }
        try {
            int[] s=interpreter.getInputTensor(0).shape();
            if(s.length!=4 || s[0]!=1 || interpreter.getInputTensor(0).dataType()!=DataType.FLOAT32)
                throw new IllegalArgumentException("Lane input must be 4D FLOAT32");
            inputNchw=s[1]==3;
            if(inputNchw){inputH=s[2];inputW=s[3];}
            else if(s[3]==3){inputH=s[1];inputW=s[2];}
            else throw new IllegalArgumentException("Lane input must be RGB NHWC or NCHW");
            if(interpreter.getOutputTensorCount()!=2) throw new IllegalArgumentException("Lane needs 2 outputs");
            int laneIndex=-1,daIndex=-1;
            for(int i=0;i<2;i++){
                String name=interpreter.getOutputTensor(i).name().toLowerCase(Locale.US);
                if(name.contains("lane") || name.contains("ll")) laneIndex=i;
                if(name.contains("drivable") || name.contains("da")) daIndex=i;
            }
            if(laneIndex<0 || daIndex<0 || laneIndex==daIndex)
                throw new IllegalArgumentException("Lane/DA output names must identify both tensors; inspect exported model tensor names");
            int[] o=interpreter.getOutputTensor(laneIndex).shape();
            if(o.length!=4 || o[0]!=1 || interpreter.getOutputTensor(laneIndex).dataType()!=DataType.FLOAT32)
                throw new IllegalArgumentException("Lane output must be 4D FLOAT32");
            outputNchw=o[1]==2;
            if(outputNchw){outputH=o[2];outputW=o[3];}
            else if(o[3]==2){outputH=o[1];outputW=o[2];}
            else throw new IllegalArgumentException("Lane output needs two channels");
            int[] da=interpreter.getOutputTensor(daIndex).shape();
            if(da.length!=4 || da[0]!=1 || (outputNchw ? da[1]!=2 || da[2]!=outputH || da[3]!=outputW : da[1]!=outputH || da[2]!=outputW || da[3]!=2))
                throw new IllegalArgumentException("DA output shape does not match lane output");
            laneOutput=outputNchw?new float[1][2][outputH][outputW]:new float[1][outputH][outputW][2];
            drivableOutput=outputNchw?new float[1][2][outputH][outputW]:new float[1][outputH][outputW][2];
            outputs.put(laneIndex,laneOutput);outputs.put(daIndex,drivableOutput);
            input=ByteBuffer.allocateDirect(inputH*inputW*3*4).order(ByteOrder.nativeOrder());
            rgbBytes=new byte[inputH*inputW*3];
            Log.i(TAG,"Lane TFLite input="+inputW+"x"+inputH+" output="+outputW+"x"+outputH);
        } catch(RuntimeException e){interpreter.close();throw e;}
    }

    public synchronized void detect(Mat frame){
        if(closed || frame==null || frame.empty()) return;
        long t0=System.nanoTime();
        int fw=frame.cols(),fh=frame.rows();
        if(lastFrameW!=fw || lastFrameH!=fh){leftTrack.clear();rightTrack.clear();lastFrameW=fw;lastFrameH=fh;frameCount=0;}
        // Python adapter: ratio=min(640/h,640/w), auto-padding modulo stride 32.
        // Scale that preprocessing to the exported TFLite tensor when it differs from 640.
        boolean run=(frameCount++ % LANE_INTERVAL)==0;
        if(run){
            double ratio=Math.min(inputW/(double)fw,inputH/(double)fh);
            int nw=Math.max(1,(int)Math.rint(fw*ratio)),nh=Math.max(1,(int)Math.rint(fh*ratio));
            int dw=(inputW-nw)%32,dh=(inputH-nh)%32;
            int left=dw/2,top=dh/2;
            // Fixed-shape TFLite requires full tensor. Preserve the centered letterbox,
            // rather than Python's dynamic auto-shape, which cannot feed a fixed tensor.
            left=(inputW-nw)/2;top=(inputH-nh)/2;
            Imgproc.resize(frame,resized,new Size(nw,nh),0,0,Imgproc.INTER_AREA);
            Core.copyMakeBorder(resized,padded,top,inputH-nh-top,left,inputW-nw-left,Core.BORDER_CONSTANT,new Scalar(114,114,114,255));
            if(padded.channels()==4) Imgproc.cvtColor(padded,rgb,Imgproc.COLOR_RGBA2RGB);
            else Imgproc.cvtColor(padded,rgb,Imgproc.COLOR_BGR2RGB);
            rgb.get(0,0,rgbBytes);input.clear();
            int plane=inputW*inputH;
            if(inputNchw){for(int c=0;c<3;c++)for(int p=0;p<plane;p++)input.putFloat((rgbBytes[3*p+c]&255)/255f);}
            else {for(byte b:rgbBytes)input.putFloat((b&255)/255f);}
            input.rewind();long t1=System.nanoTime();
            interpreter.runForMultipleInputsOutputs(new Object[]{input},outputs);
            long t2=System.nanoTime();
            // Crop output logits to the unpadded image, resize margin bilinearly,
            // then threshold. This matches the Python adapter's logit-margin order.
            float[] logits=new float[outputW*outputH];
            for(int y=0;y<outputH;y++)for(int x=0;x<outputW;x++){
                float bg=outputNchw?laneOutput[0][0][y][x]:laneOutput[0][y][x][0];
                float fg=outputNchw?laneOutput[0][1][y][x]:laneOutput[0][y][x][1];
                logits[y*outputW+x]=(!Float.isNaN(bg)&&!Float.isInfinite(bg)&&!Float.isNaN(fg)&&!Float.isInfinite(fg))?fg-bg:-1000000f;
            }
            margin.create(outputH,outputW,CvType.CV_32FC1);margin.put(0,0,logits);
            int x0=Math.max(0,Math.min(outputW-1,(int)Math.rint(left*outputW/(double)inputW)));
            int y0=Math.max(0,Math.min(outputH-1,(int)Math.rint(top*outputH/(double)inputH)));
            int x1=Math.max(x0+1,Math.min(outputW,(int)Math.rint((left+nw)*outputW/(double)inputW)));
            int y1=Math.max(y0+1,Math.min(outputH,(int)Math.rint((top+nh)*outputH/(double)inputH)));
            Mat roi=margin.submat(y0,y1,x0,x1);
            Imgproc.resize(roi,fullMargin,new Size(fw,fh),0,0,Imgproc.INTER_LINEAR);roi.release();
            byte[] mask=new byte[fw*fh];float[] row=new float[fw];maskPixels=roiPixels=0;
            int roiStart=(int)(fh*ROI_TOP),roiEnd=(int)(fh*ROI_BOTTOM);
            for(int y=0;y<fh;y++){
                fullMargin.get(y,0,row);
                for(int x=0;x<fw;x++)if(row[x]>0 && !Float.isNaN(row[x])&&!Float.isInfinite(row[x])){
                    mask[y*fw+x]=1;maskPixels++;if(y>=roiStart&&y<roiEnd)roiPixels++;
                }
            }
            extract(mask,fw,fh,t2);
            long t3=System.nanoTime();prepSum+=(t1-t0)/1e6;inferSum+=(t2-t1)/1e6;postSum+=(t3-t2)/1e6;
            if(++logFrames>=30){Log.d(TAG,String.format(Locale.US,"avg%d prep=%.1f infer=%.1f post=%.1f | mask=%d roi=%d candidates=%d lines=%d | mode=%s | %s",logFrames,prepSum/logFrames,inferSum/logFrames,postSum/logFrames,maskPixels,roiPixels,candidateCount,lineCount,virtualMode?"virtual":"detected",failure));logFrames=0;prepSum=inferSum=postSum=0;}
        }else{
            updateTrack(leftTrack,null,t0,false);updateTrack(rightTrack,null,t0,false);
            refreshPoints(fw,fh,t0);
        }
    }

    private static Line fit(List<double[]> pts,boolean[] selected){
        int n=0;double sx=0,sy=0,sxy=0,syy=0,low=1,high=0;
        for(int i=0;i<pts.size();i++)if(selected[i]){
            double x=pts.get(i)[0],y=pts.get(i)[1];n++;sx+=x;sy+=y;sxy+=x*y;syy+=y*y;low=Math.min(low,y);high=Math.max(high,y);
        }
        if(n<MIN_POINTS || high-low<MIN_SPAN)return null;
        double den=n*syy-sy*sy;if(Math.abs(den)<1e-10)return null;
        double a=(n*sxy-sy*sx)/den,b=(sx-a*sy)/n;
        if(Math.abs(a)>2.5)return null;
        Line line=new Line(a,b,low,high);return line.bottom()>=0&&line.bottom()<=1?line:null;
    }
    private void extract(byte[] mask,int fw,int fh,long now){
        // Python extractor downsamples only if source width exceeds 640.
        int sw=Math.min(fw,640),sh=Math.max(1,(int)Math.rint(fh*(sw/(double)fw)));
        byte[] small;
        if(sw==fw){small=mask;}else{
            Mat src=new Mat(fh,fw,CvType.CV_8UC1),dst=new Mat();src.put(0,0,mask);
            Imgproc.resize(src,dst,new Size(sw,sh),0,0,Imgproc.INTER_NEAREST);
            small=new byte[sw*sh];dst.get(0,0,small);src.release();dst.release();
        }
        ArrayList<double[]> pts=new ArrayList<>();
        for(int y=(int)(sh*ROI_TOP);y<(int)(sh*ROI_BOTTOM);y+=Math.max(1,sh/100)){
            int x=0;while(x<sw){if(small[y*sw+x]==0){x++;continue;}int begin=x;while(x<sw&&small[y*sw+x]!=0)x++;int count=x-begin;
                if(count>0&&count<=sw*.45)pts.add(new double[]{(begin+(x-1))*.5/sw,y/(double)sh});
            }
        }
        candidateCount=pts.size();ArrayList<Line> lines=new ArrayList<>();Random rng=new Random(42);
        for(int k=0;k<8&&pts.size()>=MIN_POINTS;k++){
            boolean[] best=null;int bestCount=0;
            for(int trial=0;trial<100;trial++){
                int i=rng.nextInt(pts.size()),j=rng.nextInt(pts.size()-1);if(j>=i)j++;
                double[] p=pts.get(i),q=pts.get(j);if(Math.abs(p[1]-q[1])<MIN_SPAN)continue;
                double a=(p[0]-q[0])/(p[1]-q[1]);if(Math.abs(a)>2.5)continue;
                double b=p[0]-a*p[1];boolean[] selected=new boolean[pts.size()];int count=0;
                for(int z=0;z<pts.size();z++){double[] v=pts.get(z);if(Math.abs(v[0]-a*v[1]-b)<RESIDUAL){selected[z]=true;count++;}}
                if(count>bestCount){best=selected;bestCount=count;}
            }
            if(best==null||bestCount<MIN_POINTS)break;
            Line l=fit(pts,best);ArrayList<double[]> remaining=new ArrayList<>();
            for(int i=0;i<pts.size();i++)if(!best[i])remaining.add(pts.get(i));pts=remaining;
            if(l!=null)lines.add(l);
        }
        lineCount=lines.size();Line l=null,r=null;
        for(Line v:lines){if(v.bottom()<CENTER&&(l==null||v.bottom()>l.bottom()))l=v;
            if(v.bottom()>CENTER&&(r==null||v.bottom()<r.bottom()))r=v;}
        if(l!=null&&r!=null){double width=r.bottom()-l.bottom(),top=Math.max(l.low,r.low);
            if(width<MIN_WIDTH||width>MAX_WIDTH||r.x(top)<=l.x(top)){l=null;r=null;failure="pair_rejected";}}
        else failure=roiPixels==0?"empty_roi_mask":"no_valid_ego_pair";
        if(l!=null||r!=null)failure="accepted";
        updateTrack(leftTrack,l,now,true);updateTrack(rightTrack,r,now,true);
        refreshPoints(fw,fh,now);
    }
    private void updateTrack(SideTrack t,Line observed,long now,boolean detectionRan){
        if(t.current!=null&&(now-t.lastSeenNs)/1e9>HOLD_SECONDS)t.clear();
        if(observed!=null){
            if((t==leftTrack&&observed.x(.98)>=CENTER)||(t==rightTrack&&observed.x(.98)<=CENTER))observed=null;
            else if(t.current!=null){double jump=Math.max(Math.abs(observed.x(.65)-t.current.x(.65)),Math.abs(observed.x(.9)-t.current.x(.9)));
                if(jump>MAX_JUMP)observed=null;}
        }
        if(observed!=null){
            if(t.current==null){t.current=observed.copy();t.hits=1;t.confidence=.75;}
            else{t.hits++;t.confidence=Math.min(.95,t.confidence+.15);}
            t.target=observed.copy();t.lastObserved=observed.copy();t.lastSeenNs=now;t.valid=true;
        }else if(detectionRan)t.valid=false;
        if(t.current!=null&&t.target!=null){Line a=t.current,b=t.target;
            t.current=new Line(ALPHA*b.a+(1-ALPHA)*a.a,ALPHA*b.b+(1-ALPHA)*a.b,ALPHA*b.low+(1-ALPHA)*a.low,ALPHA*b.high+(1-ALPHA)*a.high);}
    }
    private static void append(List<Point> dst,Line l,double low,double high,int w,int h){
        for(int i=0;i<SAMPLES;i++){double y=low+(high-low)*i/(SAMPLES-1.0),x=l.x(y);
            if(x>=0&&x<1&&y>=0&&y<1)dst.add(new Point(Math.min(w-1,Math.round(x*w)),Math.min(h-1,Math.round(y*h))));}
    }
    private static double projectedHigh(double low,double high){
        double span=Math.max(0,high-low);return span>=.001?Math.min(.98,high+Math.max(0,.50-span)):high;
    }
    private static double projectedLow(double low) {
        return Math.max(0.30, low - 0.35);
    }
    private void addTrack(SideTrack t,List<Point> display,List<Point> reliable,int w,int h,long now){
        if(t.current==null||t.hits<1)return;
        Line l=t.current;append(display,l,l.low,projectedHigh(l.low,l.high),w,h);
        double age=Math.max(0,(now-t.lastSeenNs)/1e9);
        double confidence=t.confidence*Math.max(.2,1-(t.valid?.2:.8)*age/Math.max(HOLD_SECONDS,1e-9));
        // Classification requires a fresh, confidence-gated observation.
        // On skipped frames, freshness is bounded by the lane sampling interval.
        if(t.valid&&age<=HOLD_SECONDS&&confidence>=.65&&t.lastObserved!=null){
            Line obs=t.lastObserved;
            append(reliable, l, projectedLow(obs.low),
                    projectedHigh(obs.low, obs.high), w, h);
        }
    }
    private void refreshPoints(int w,int h,long now){
        displayLeft.clear();displayRight.clear();reliableLeft.clear();reliableRight.clear();
        if(leftTrack.current!=null&&rightTrack.current!=null){
            Line l=leftTrack.current,r=rightTrack.current;
            double low=Math.max(l.low,r.low),high=Math.min(l.high,r.high);
            if(low>=high||r.x(low)-l.x(low)<=.015||r.x(high)-l.x(high)<=.015){failure="tracked_pair_invalid";}
            else{addTrack(leftTrack,displayLeft,reliableLeft,w,h,now);addTrack(rightTrack,displayRight,reliableRight,w,h,now);}
        }else{addTrack(leftTrack,displayLeft,reliableLeft,w,h,now);addTrack(rightTrack,displayRight,reliableRight,w,h,now);}
        // Python pipeline: a real lane requires both sides; otherwise a virtual pair.
        virtualMode=reliableLeft.size()<2||reliableRight.size()<2;
        if(virtualMode){
            displayLeft.clear();displayRight.clear();reliableLeft.clear();reliableRight.clear();
            for(int i=0;i<SAMPLES;i++){
                double t=i/(double)(SAMPLES-1),y=.40+.60*t;
                int py=Math.min(h-1,(int)Math.round(y*h));
                Point l=new Point(Math.min(w-1,Math.round((.48-.32*t)*w)),py);
                Point r=new Point(Math.min(w-1,Math.round((.52+.32*t)*w)),py);
                displayLeft.add(l);displayRight.add(r);reliableLeft.add(l);reliableRight.add(r);
            }
        }
    }
    public synchronized void invalidate(){leftTrack.clear();rightTrack.clear();displayLeft.clear();displayRight.clear();reliableLeft.clear();reliableRight.clear();failure="inference_error";virtualMode=false;}
    // Python: LANE_FILL_BGR=(255,220,155), alpha=0.25.
    // CameraActivity passes RGBA, so the equivalent Scalar is (155,220,255,255).
    private static final Scalar LANE_FILL_RGBA = new Scalar(155, 220, 255, 255);
    private static final double LANE_FILL_ALPHA = 0.25;

    private static double interpolateX(List<Point> points, double y) {
        if (y <= points.get(0).y) return points.get(0).x;
        for (int i = 1; i < points.size(); i++) {
            Point a = points.get(i - 1), b = points.get(i);
            if (y <= b.y) {
                double dy = b.y - a.y;
                return Math.abs(dy) < 1e-9 ? b.x :
                        a.x + (y - a.y) * (b.x - a.x) / dy;
            }
        }
        return points.get(points.size() - 1).x;
    }

    public synchronized void drawLastResult(Mat frame) {
        if (closed || frame == null || frame.empty()) return;
        if (displayLeft.size() < 2 || displayRight.size() < 2) return;

        // Work on copies so the points used by YOLO are unchanged.
        ArrayList<Point> left = new ArrayList<>(displayLeft);
        ArrayList<Point> right = new ArrayList<>(displayRight);
        Collections.sort(left, (a, b) -> Double.compare(a.y, b.y));
        Collections.sort(right, (a, b) -> Double.compare(a.y, b.y));

        double lo = Math.max(left.get(0).y, right.get(0).y);
        double hi = Math.min(left.get(left.size() - 1).y,
                right.get(right.size() - 1).y);
        if (hi <= lo) return;

        final int n = 60, w = frame.cols(), h = frame.rows();
        Point[] polygon = new Point[n * 2];
        for (int i = 0; i < n; i++) {
            double y = lo + (hi - lo) * i / (n - 1.0);
            double lx = interpolateX(left, y);
            double rx = interpolateX(right, y);
            if (rx <= lx) return; // Same non-crossing check as Python.
            double cy = Math.max(0, Math.min(h - 1, Math.round(y)));
            polygon[i] = new Point(Math.max(0, Math.min(w - 1, Math.round(lx))), cy);
            polygon[2 * n - 1 - i] =
                    new Point(Math.max(0, Math.min(w - 1, Math.round(rx))), cy);
        }

        Mat overlay = frame.clone();
        MatOfPoint polygonMat = new MatOfPoint();
        polygonMat.fromArray(polygon);
        int[] coords = new int[polygon.length * 2];
        for (int i = 0; i < polygon.length; i++) {
            coords[2 * i] = (int) polygon[i].x;
            coords[2 * i + 1] = (int) polygon[i].y;
        }
        polygonMat.put(0, 0, coords);
        try {
            Imgproc.fillPoly(overlay, Collections.singletonList(polygonMat),
                    LANE_FILL_RGBA);
            Core.addWeighted(overlay, LANE_FILL_ALPHA, frame,
                    1.0 - LANE_FILL_ALPHA, 0.0, frame);
        } finally {
            polygonMat.release();
            overlay.release();
        }
    }

    public synchronized List<Point> getLeftPoints(){return new ArrayList<>(reliableLeft);}
    public synchronized List<Point> getRightPoints(){return new ArrayList<>(reliableRight);}
    public synchronized List<Point> getDisplayLeftPoints() {
        return new ArrayList<>(displayLeft);
    }

    public synchronized List<Point> getDisplayRightPoints() {
        return new ArrayList<>(displayRight);
    }
    public synchronized int getMaskPixels(){return maskPixels;}
    public synchronized int getRoiPixels(){return roiPixels;}
    public synchronized int getLineCount(){return lineCount;}
    public synchronized String getFailure(){return virtualMode?"virtual / "+failure:failure;}
    public synchronized int getDisplayLeftCount(){return displayLeft.size();}
    public synchronized int getDisplayRightCount(){return displayRight.size();}
    @Override public synchronized void close(){if(closed)return;closed=true;interpreter.close();resized.release();padded.release();rgb.release();margin.release();cropped.release();fullMargin.release();sceneGray.release();thumbnail.release();}
}
