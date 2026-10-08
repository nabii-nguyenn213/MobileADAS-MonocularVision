import torch
from ai.src.lane_detection.models.backbone import ResNetBackbone


class UFLDv2(torch.nn.Module):
    def __init__(self, backbone="resnet18", num_grid_row=100, num_grid_col=100, num_row=56, num_col=41,
                 num_lanes=4, input_height=320, input_width=800, hidden_dim=2048):
        super().__init__()
        self.num_grid_row = num_grid_row
        self.num_grid_col = num_grid_col
        self.num_row = num_row
        self.num_col = num_col
        self.num_lanes = num_lanes
        self.input_height = input_height
        self.input_width = input_width

        self.model = ResNetBackbone(backbone=backbone, pretrained=False)

        self.pool = torch.nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=1,
        )

        self.feature_height = input_height // 32
        self.feature_width = input_width // 32

        self.input_dim = (
            8
            * self.feature_height
            * self.feature_width
        )

        self.loc_row_dim = (
            num_grid_row
            * num_row
            * num_lanes
        )

        self.loc_col_dim = (
            num_grid_col
            * num_col
            * num_lanes
        )

        self.exist_row_dim = (
            2
            * num_row
            * num_lanes
        )

        self.exist_col_dim = (
            2
            * num_col
            * num_lanes
        )

        self.total_dim = (
            self.loc_row_dim
            + self.loc_col_dim
            + self.exist_row_dim
            + self.exist_col_dim
        )

        self.hidden_dim = hidden_dim
        self.cls = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(
                self.input_dim,
                hidden_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden_dim,
                self.total_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        features = self.model(x)

        feature = features[-1]

        feature = self.pool(feature)

        feature = torch.flatten(
            feature,
            start_dim=1,
        )

        output = self.cls(feature)

        predictions = {

            "loc_row": output[
                :, :self.loc_row_dim
            ].view(
                -1,
                self.num_grid_row,
                self.num_row,
                self.num_lanes,
            ),

            "loc_col": output[
                :,
                self.loc_row_dim:
                self.loc_row_dim + self.loc_col_dim
            ].view(
                -1,
                self.num_grid_col,
                self.num_col,
                self.num_lanes,
            ),

            "exist_row": output[
                :,
                self.loc_row_dim + self.loc_col_dim:
                self.loc_row_dim
                + self.loc_col_dim
                + self.exist_row_dim
            ].view(
                -1,
                2,
                self.num_row,
                self.num_lanes,
            ),

            "exist_col": output[
                :, -self.exist_col_dim:
            ].view(
                -1,
                2,
                self.num_col,
                self.num_lanes,
            ),
        }
        return predictions

if __name__ == "__main__":
    model = UFLDv2()

    x = torch.randn(1, 3, 320, 800)

    pred = model(x)

    for name, value in pred.items():
        print(name, value.shape)
