import os

import torch


class LaneModel:

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.model.eval()

    def load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:

        if not os.path.exists(
            checkpoint_path
        ):
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # --------------------------------
        # Official UFLDv2 checkpoints
        # usually store weights under:
        #
        # checkpoint["model"]
        # --------------------------------

        if (
            isinstance(checkpoint, dict)
            and "model" in checkpoint
        ):
            state_dict = checkpoint["model"]

        else:
            # Also support checkpoint files
            # containing only state_dict.
            state_dict = checkpoint

        # --------------------------------
        # Remove DistributedDataParallel
        # prefix:
        #
        # module.model.conv1...
        # ->
        # model.conv1...
        # --------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith(
                "module."
            ):
                key = key[
                    len("module.") :
                ]

            cleaned_state_dict[
                key
            ] = value

        # --------------------------------
        # Strict=True is intentional.
        #
        # If architecture and checkpoint
        # do not match, we want to know
        # instead of silently ignoring it.
        # --------------------------------

        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def predict(
        self,
        input_tensor: torch.Tensor,
    ):
        if input_tensor is None:
            raise ValueError(
                "Input tensor cannot be None."
            )

        input_tensor = input_tensor.to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():

            output = self.model(
                input_tensor
            )

        return output
