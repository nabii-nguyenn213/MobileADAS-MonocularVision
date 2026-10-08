import torch 

class LaneCheckpointLoader: 
    def __init__(self, checkpoint_path, device="cpu"): 
        self.checkpoint_path = checkpoint_path
        self.device = device

    def load(self, model:torch.nn.Module): 
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"]
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."): 
                key = key[7:]
            cleaned_state_dict[key]=value

        model.load_state_dict(cleaned_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model
