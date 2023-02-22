import time
import urllib.request
from multiprocessing import Pool, Process, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import splitfolders
from PIL import Image


class Database:

    def __init__(self, csv_file: str, imgs_threshold=100, database_name="Database"):
        # Directories Variables
        self.cwd = Path.cwd()
        self.main_folder = self.cwd.parent
        self.observations_name = "Observations"
        self.observations_path = self.main_folder / self.observations_name

        # Constructor Variables
        self.csv_file = csv_file
        self.csv_file_path = self.observations_path / self.csv_file
        self.imgs_threshold = imgs_threshold

        # Database Variables
        self.database_name = database_name
        self.database_path = self.main_folder / self.database_name

        # String Variables
        self.id = "id"
        self.license = "license"
        self.img_url = "image_url"
        self.scientific_name = "scientific_name"
        self.plot_name = "distribution.png"

    def create_database(self, print_info=False, plot_graph=False, n_cpus=cpu_count()):
        # Info dict
        info = {}

        # Load CSV
        df = pd.read_csv(self.csv_file_path)
        info['init_imgs'] = len(df)

        # Remove missing values
        df = df.dropna()
        info['drop_na_imgs'] = len(df)

        # Remove species below threshold
        df = df.groupby(self.scientific_name).filter(lambda x: len(x) >= self.imgs_threshold)
        info['filtered_imgs'] = len(df)

        # Create the database
        species_index = sorted(df[self.scientific_name].unique())
        species_groups = df.groupby(self.scientific_name)
        self.create_species_directories(species_index)
        self.save_images(species_groups, n_cpus)

        # Info
        if print_info:
            info['num_classes'] = len(species_index)
            print(f"Qtde de imagens inicialmente: {info['init_imgs']}")
            print(f"Qtde de imagens após dropna: {info['drop_na_imgs']}")
            print(f"Qtde de imagens após filtragem final: {info['filtered_imgs']}")
            print(f"Qtde de classes: {info['num_classes']}")

        # Class distribution plot
        if plot_graph:
            self.plot_distribution_graph(df)

    def create_species_directories(self, species_index):
        for species_name in species_index:
            dir_location = self.database_path / species_name
            self.create_directory(dir_location)

    def save_images(self, species_groups, n_cpus):
        processes = []

        # Parallelize download
        for _, species_group in species_groups:
            num_images = len(species_group)
            split_size = num_images // n_cpus

            # Create processes
            for i in range(n_cpus):
                start = i * split_size
                end = (i+1) * split_size if (i + 1) != n_cpus else num_images
                new_process = Process(target=self.download_images, args=(species_group, start, end))
                processes.append(new_process)

        # Start processes
        for process in processes:
            process.start()

    def download_images(self, species_group, start, end):
        species_group = species_group.reset_index()  # Reset index to access rows sequentially
        valid_img_formats = {".png", ".jpg", ".jpeg"}

        # Group columns
        ids = species_group[self.id]
        urls = species_group[self.img_url]
        species_name = species_group[self.scientific_name][0]
        species_image_file_name = species_name.replace(" ", "_").lower()

        for i in range(start, end):
            # Check file extension
            file_ext = Path(urls[i]).suffix
            if file_ext.lower() not in valid_img_formats:
                continue

            # Give each image a name
            img_name = f"{species_image_file_name}_{i}_{ids[i]}" + file_ext
            img_location = self.database_path / species_name / img_name

            # Try to download the image
            if not img_location.exists():
                self.save_img_from_url(url=urls[i], img_location=img_location)

    def check_images(self):
        for img_dir in self.database_path.iterdir():
            for img in img_dir.iterdir():
                self.check_image(img)
        print("Checked all images.")

    def split_dataset(self,
                      dataset_name="Dataset",
                      ratio_split=None,
                      fixed_split=None,
                      oversample=False,
                      split_seed=1337,
                      move=False):
        """ Split image database into Train/Validation/(Test) using 'splitfolders' package"""
        input_path = self.database_path
        output_path = self.main_folder / dataset_name

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

    def plot_distribution_graph(self, df):
        species_distribution = df[self.scientific_name].value_counts().sort_values(ascending=False)
        species_distribution.plot(kind='bar', figsize=(10, 10))
        plt.grid(axis='y', linestyle='--', color='grey')
        plt.subplots_adjust(bottom=0.25)

        plt.xlabel("Nome Científico")
        plt.ylabel("Quantidade de Imagens")
        plt.title("Quantidade de imagens por espécie")
        plt.savefig(self.observations_path / self.plot_name)

    @staticmethod
    def create_directory(dir_location):
        # Create missing parents as needed
        # Don't raise errors if the directory already exists
        Path(dir_location).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_img_from_url(url, img_location):
        # Try to download image
        # If not successful, pause a little before trying again
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, img_location)
            except Exception as e:
                print(f"{url}: {e}")
                time.sleep(attempt)
            else:
                return
        print(f"Skipping {url}")

    @staticmethod
    def check_image(image):
        # Check if image can be opened
        # Verify if image is not broken
        try:
            with Image.open(image) as img:
                img.verify()
        except Exception as e:
            print(f"Image '{image.name}': {e}")
            print(f"Removed '{image.name}'")
            Path.unlink(image)


def main():
    db = Database(csv_file="spiders.csv", imgs_threshold=100)
    db.create_database(print_info=False, plot_graph=False)
    # db.check_images()
    # db.split_dataset(dataset_name="Dataset", fixed_split=60, oversample=True, split_seed=17823)


if __name__ == "__main__":
    main()
