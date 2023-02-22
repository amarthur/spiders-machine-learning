import time
import urllib.request
from multiprocessing import Process, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import splitfolders
from PIL import Image
from tqdm import tqdm


class Database:

    def __init__(self, csv_file: str, database_name="Database", imgs_threshold=100):
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

    def create_database(self, check_images=True, print_info=False, plot_graph=False, n_cpus=cpu_count()):
        # Info dict
        info = {}

        # Load CSV
        df = pd.read_csv(self.csv_file_path)
        info["Nº imgs initially"] = len(df)

        # Remove missing values
        df = df.dropna()
        info["Nº imgs after removing missing values"] = len(df)

        # Remove invalid formats
        valid_img_formats = [".png", ".jpg", ".jpeg"]
        df = df[df[self.img_url].apply(lambda x: Path(x).suffix.lower() in valid_img_formats)]

        # Remove species below threshold
        df = df.groupby(self.scientific_name).filter(lambda x: len(x) >= self.imgs_threshold)
        info["Nº imgs after removing species below threshold"] = len(df)

        # Create the database
        species_index = sorted(df[self.scientific_name].unique())
        species_groups = df.groupby(self.scientific_name)
        info["Nº classes"] = len(species_index)

        print("Creating directories...")
        self.create_species_directories(species_index)
        print("Finished creating directories.\n")

        print("Downloading images...")
        self.save_images(species_groups, n_cpus)
        print("Finished downloading images.\n")

        if check_images:
            self.check_images()

        if print_info:
            self.print_info(info)

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

        # Join processes
        for process in processes:
            process.join()

    def download_images(self, species_group, start, end):
        species_group = species_group.reset_index()  # Reset index to access rows sequentially

        # Group columns
        ids = species_group[self.id]
        urls = species_group[self.img_url]
        species_name = species_group[self.scientific_name][0]
        species_image_file_name = species_name.replace(" ", "_").lower()

        for i in range(start, end):
            # Give each image a name
            file_ext = Path(urls[i]).suffix
            img_name = f"{species_image_file_name}_{i}_{ids[i]}" + file_ext
            img_location = self.database_path / species_name / img_name

            # Try to download the image
            if not img_location.exists():
                self.save_img_from_url(url=urls[i], img_location=img_location)

    def check_images(self):
        images = [img for img_dir in self.database_path.iterdir() for img in img_dir.iterdir()]
        for image in tqdm(images, desc="Checking images", unit=" images"):
            self.check_image(image)

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
    def print_info(info):
        print("\nInfo:")
        for k, v in info.items():
            print(f"{k}: {v}")
        print()

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
            print(f"Image '{image}': {e}")
            print(f"Removed '{image}'")
            Path.unlink(image)


def main():
    db = Database(csv_file="spiders.csv", imgs_threshold=100)
    db.create_database(check_images=True, print_info=True, plot_graph=False)
    db.split_dataset(fixed_split=60, oversample=False, split_seed=17823)


if __name__ == "__main__":
    main()
