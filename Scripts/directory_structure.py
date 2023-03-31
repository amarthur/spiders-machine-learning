from pathlib import Path


class DirectoryStructure:

    def __init__(self, database_dir_name=None, dataset_dir_name=None, test=False) -> None:
        # Parameters
        self.database_dir_name = database_dir_name
        self.dataset_dir_name = dataset_dir_name
        self.test = test

        # Naming Variables
        self.scripts_dir_name = "Scripts"
        self.observations_dir_name = "Observations"
        self.weights_dir_name = "SavedWeights"

        self.train_phase = "train"
        self.valid_phase = "val"
        self.test_phase = "test"

        # Directories
        self.main_dir = Path.cwd().parent
        self.scripts_dir = self.main_dir / self.scripts_dir_name
        self.observations_dir = self.main_dir / self.observations_dir_name
        self.weights_dir = self.main_dir / self.weights_dir_name

        self.phases = [self.train_phase, self.valid_phase]
        if self.test:
            self.phases.append(self.test_phase)

        if database_dir_name is not None:
            self.database_dir = self.main_dir / self.databases_dir_name / database_dir_name

        if dataset_dir_name is not None:
            self.dataset_dir = self.main_dir / self.datasets_dir_name / dataset_dir_name
            self.phases_dirs = {phase: self.dataset_dir / phase for phase in self.phases}
            self.phases_csv = {phase: f"{phase_dir/phase}.csv" for phase, phase_dir in self.phases_dirs.items()}

    def database_dict(self) -> dict:
        """Returns a python dict with key-value pairs as `{Path(class_directory): list(Path(images)}`"""
        return {
            class_dir: [image for image in class_dir.iterdir() if Path.is_file(image)
                       ] for class_dir in self.database_dir.iterdir() if Path.is_dir(class_dir)
        } if Path.is_dir(self.database_dir) else {}

    def dataset_dict(self) -> dict:
        """Returns a python dict with key-value pairs as `{Path(class_directory): list(Path(images)}`
        for each phase in the dataset"""
        return {
            phase: {
                class_dir: [image for image in class_dir.iterdir() if Path.is_file(image)
                           ] for class_dir in phase_dir.iterdir() if Path.is_dir(class_dir)
            } for phase, phase_dir in self.phases_dirs.items() if Path.is_dir(phase_dir)
        }


def main():
    return


if __name__ == "__main__":
    main()
