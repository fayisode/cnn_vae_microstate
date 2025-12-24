import toml


class Config:
    def __init__(self, target_file):
        self.file = target_file
        self.load_toml_config()

    def load_toml_config(self):
        with open(self.file, "r") as file:
            self.config = toml.load(file)

    def get_model_config(self):
        return self.config.get("vae", {})

    def get_eeg_config(self):
        return self.config.get("eeg", {})

    def get_test_info(self):
        return self.config.get("data", {}).get("test_url", None), self.config.get(
            "data", {}
        ).get("test_dir", None)

    def get_train_info(self):
        return self.config.get("data", {}).get("train_url", None), self.config.get(
            "data", {}
        ).get("train_dir", None)

    def get_batch_size(self):
        return self.get_model_config().get("batch_size", 0)


config = Config("./config/config.toml")
