import time
import urllib.request
from multiprocessing.pool import ThreadPool
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from directory_structure import DirectoryStructure
from PIL import Image


class Database:

    def __init__(self, csv_file: str, database_name="Database", imgs_threshold=100, processes=64):
        # Parameters
        self.csv_file = csv_file
        self.database_name = database_name
        self.imgs_threshold = imgs_threshold
        self.processes = processes

        # Directories
        self.dirs = DirectoryStructure(database_dir_name=database_name)
        self.csv_file_path = self.dirs.observations_dir / self.csv_file

        # CSV String Variables
        self.id = "id"
        self.license = "license"
        self.img_url = "image_url"
        self.img_name = "image_name"
        self.img_path = "image_path"
        self.scientific_name = "scientific_name"

    def create_database(self,
                        save_images: bool = False,
                        check_images: bool = True,
                        print_info: bool = False,
                        database_csv_name: str = None,
                        plot_name: str = None):
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

        # Create directories
        species_index = sorted(df[self.scientific_name].unique())
        info["Nº classes"] = len(species_index)

        print("Creating directories...")
        self.create_species_directories(species_index)
        print("Finished creating directories.\n")

        # Download images
        species_groups = df.groupby(self.scientific_name)
        groups_df = [self.define_images(species_group) for _, species_group in species_groups]
        images_df = pd.concat(groups_df, axis=0)

        if save_images:
            print("Downloading images...")
            self.save_images(df=images_df)
            print("Finished downloading images.\n")

        if check_images:
            paths = [path for path in images_df[self.img_path]]
            self.check_images(paths)

        if print_info:
            self.print_info(info)

        if database_csv_name:
            self.create_csv(images_df, database_csv_name)

        if plot_name:
            self.plot_distribution_graph(df, plot_name)

    def create_species_directories(self, species_index):
        for species_name in species_index:
            dir_location = self.dirs.database_dir / species_name
            self.create_directory(dir_location)

    def define_images(self, species_group):
        # Get group data
        group_data = species_group.reset_index()
        i = group_data.index.astype(str)
        ext = group_data[self.img_url].apply(lambda x: Path(x).suffix)

        # Naming
        species_name = group_data[self.scientific_name][0]
        image_name = species_name.replace(" ", "_")

        # Define image names and paths
        group_data[self.img_name] = image_name + "_" + i + ext
        group_data[self.img_path] = group_data[self.img_name].apply(lambda x: self.dirs.database_dir / species_name / x)
        return group_data

    def save_images(self, df):
        urls_paths = [(url, path) for url, path in zip(df[self.img_url], df[self.img_path])]
        self.multiprocess(self.download_image, urls_paths)

    def check_images(self, paths):
        self.multiprocess(self.check_image, paths)

    def multiprocess(self, func, func_args):
        with ThreadPool(processes=self.processes) as pool:
            pool.starmap_async(func, func_args)
            pool.close()
            pool.join()

    def create_csv(self, df, database_csv_name):
        database_csv_path = self.dirs.database_dir / database_csv_name
        columns = [self.id, self.license, self.img_name, self.scientific_name, self.img_url]
        df.to_csv(database_csv_path, columns=columns, index=False)

    def plot_distribution_graph(self, df, plot_name):
        species_distribution = df[self.scientific_name].value_counts().sort_values(ascending=False)
        species_distribution.plot(kind='bar', figsize=(10, 10))
        plt.grid(axis='y', linestyle='--', color='grey')
        plt.subplots_adjust(bottom=0.25)

        plt.xlabel("Nome Científico")
        plt.ylabel("Quantidade de Imagens")
        plt.title("Quantidade de imagens por espécie")
        plt.savefig(self.dirs.observations_dir / plot_name)

    @staticmethod
    def print_info(info):
        print("\nInfo:")
        for info_description, info_value in info.items():
            print(f"{info_description}: {info_value}")
        print()

    @staticmethod
    def create_directory(dir_location):
        Path(dir_location).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def download_image(url, img_path):
        if img_path.exists():
            return

        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, img_path)
            except Exception as e:
                print(f"{url}: {e}")
                time.sleep(attempt)
            else:
                return
        print(f"Skipping {url}")

    @staticmethod
    def check_image(image):
        try:
            with Image.open(image) as img:
                img.verify()
        except Exception as e:
            print(f"Image '{image}': {e}")


def main():
    db = Database(csv_file="spiders.csv", imgs_threshold=100)
    db.create_database(save_images=True,
                       check_images=True,
                       print_info=True,
                       database_csv_name="spiders_database.csv",
                       plot_name="distribution.png")


if __name__ == "__main__":
    main()
