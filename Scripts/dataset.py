import pandas as pd
import splitfolders
from directory_structure import DirectoryStructure


class Dataset:

    def __init__(self, database_name="Database", dataset_name="Dataset") -> None:
        # Parameters
        self.database_name = database_name
        self.dataset_name = dataset_name

        # Directories
        self.dirs = DirectoryStructure(database_dir_name=database_name, dataset_dir_name=dataset_name)
        self.phases = self.dirs.phases

        # CSV Variables
        self.image_header = "image"
        self.class_header = "class"
        self.phases_csv = self.dirs.phases_csv

    def create_dataset(self,
                       ratio_split=None,
                       fixed_split=None,
                       oversample=False,
                       split_seed=1337,
                       move=False,
                       create_csv=True):

        input_path = self.dirs.database_dir
        output_path = self.dirs.dataset_dir

        if ratio_split is not None:
            splitfolders.ratio(input=input_path,
                               output=output_path,
                               seed=split_seed,
                               ratio=ratio_split,
                               group_prefix=None,
                               move=move)

        elif fixed_split is not None:
            splitfolders.fixed(input=input_path,
                               output=output_path,
                               seed=split_seed,
                               fixed=fixed_split,
                               oversample=oversample,
                               group_prefix=None,
                               move=move)

        if create_csv:
            self.create_dataset_csv()

    def create_dataset_csv(self):
        for phase, phase_data in self.dirs.dataset_dict().items():
            data = [(img.name, class_dir.name) for class_dir, images in phase_data.items() for img in images]
            df = pd.DataFrame(data, columns=[self.image_header, self.class_header])
            df.to_csv(self.phases_csv[phase], index=False)


def main():
    ds = Dataset()


if __name__ == "__main__":
    main()
