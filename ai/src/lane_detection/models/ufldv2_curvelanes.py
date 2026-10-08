import torch
from ai.src.lane_detection.models.backbone import ResNetBackbone

class UFLDv2CurveLanes(torch.nn.Module):

    def __init__(
        self,
        backbone: str = "resnet34",
        num_grid_row: int = 200,
        num_grid_col: int = 100,
        num_row: int = 72,
        num_col: int = 81,
        num_lanes: int = 10,
        input_height: int = 800,
        input_width: int = 1600,
        hidden_dim: int = 2048,
    ):
        super().__init__()

        self.num_grid_row = num_grid_row
        self.num_grid_col = num_grid_col

        self.num_row = num_row
        self.num_col = num_col

        self.num_lanes = num_lanes

        self.input_height = input_height
        self.input_width = input_width

        # ---------------------------------
        # Dimensions for one lane token
        # ---------------------------------

        self.dim1 = (
            num_grid_row
            * num_row
        )

        self.dim2 = (
            2
            * num_row
        )

        self.dim3 = (
            num_grid_col
            * num_col
        )

        self.dim4 = (
            2
            * num_col
        )

        self.total_dim_row = (
            self.dim1
            + self.dim2
        )

        self.total_dim_col = (
            self.dim3
            + self.dim4
        )

        # 8 channels from pool
        # +
        # 1 lane-token channel
        #
        # = 9

        self.input_dim = (
            (input_height // 32)
            * (input_width // 32)
            * 9
        )

        # ---------------------------------
        # Backbone
        # ---------------------------------

        self.model = ResNetBackbone(
            backbone=backbone,
            pretrained=False,
        )

        # ---------------------------------
        # Lane-token generation
        # ---------------------------------

        self.cls_distribute = torch.nn.Sequential(
            torch.nn.Conv2d(
                512,
                128,
                kernel_size=3,
                padding=1,
            ),

            torch.nn.ReLU(),

            torch.nn.Conv2d(
                128,
                20,
                kernel_size=3,
                padding=1,
            ),
        )

        # ---------------------------------
        # Shared lane-token MLP
        # ---------------------------------

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(
                self.input_dim
            ),

            torch.nn.Linear(
                self.input_dim,
                hidden_dim,
            ),

            torch.nn.ReLU(),
        )

        # Separate row / column heads

        self.cls_row = torch.nn.Linear(
            hidden_dim,
            self.total_dim_row,
        )

        self.cls_col = torch.nn.Linear(
            hidden_dim,
            self.total_dim_col,
        )

        # Backbone feature compression
        self.pool = torch.nn.Conv2d(
            512,
            8,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        features = self.model(x)

        feature = features[-1]

        batch_size = feature.shape[0]

        feature_height = (
            self.input_height // 32
        )

        feature_width = (
            self.input_width // 32
        )

        # ---------------------------------
        # Generate 20 lane tokens
        # ---------------------------------

        lane_token = self.cls_distribute(
            feature
        )

        lane_token = lane_token.reshape(
            batch_size,
            20,
            1,
            feature_height,
            feature_width,
        )

        # ---------------------------------
        # Compress backbone feature
        # ---------------------------------

        feature = self.pool(feature)

        feature = feature.unsqueeze(1)

        feature = feature.repeat(
            1,
            20,
            1,
            1,
            1,
        )

        # 8 backbone channels + 1 lane token
        feature = torch.cat(
            [
                feature,
                lane_token,
            ],
            dim=2,
        )

        feature = feature.reshape(
            -1,
            self.input_dim,
        )

        # ---------------------------------
        # Shared classifier
        # ---------------------------------

        output = self.cls(feature)

        output = output.reshape(
            batch_size,
            20,
            -1,
        )

        # First 10 tokens = row
        output_row = self.cls_row(
            output[:, :10, :]
        )

        # Last 10 tokens = column
        output_col = self.cls_col(
            output[:, 10:, :]
        )

        output_row = output_row.permute(
            0,
            2,
            1,
        )

        output_col = output_col.permute(
            0,
            2,
            1,
        )

        predictions = {

            "loc_row": output_row[
                :,
                :self.dim1,
                :
            ].reshape(
                -1,
                self.num_grid_row,
                self.num_row,
                self.num_lanes,
            ),

            "loc_col": output_col[
                :,
                :self.dim3,
                :
            ].reshape(
                -1,
                self.num_grid_col,
                self.num_col,
                self.num_lanes,
            ),

            "exist_row": output_row[
                :,
                self.dim1:
                self.dim1 + self.dim2,
                :
            ].reshape(
                -1,
                2,
                self.num_row,
                self.num_lanes,
            ),

            "exist_col": output_col[
                :,
                self.dim3:
                self.dim3 + self.dim4,
                :
            ].reshape(
                -1,
                2,
                self.num_col,
                self.num_lanes,
            ),

            "lane_token_row": (
                lane_token[
                    :, :10, :, :, :
                ]
                .sum(dim=1)
            ),

            "lane_token_col": (
                lane_token[
                    :, 10:, :, :, :
                ]
                .sum(dim=1)
            ),
        }

        return predictions
