import os
import pandas as pd
import pymysql
from dotenv import load_dotenv

load_dotenv()


class LoadData:
    """
    Handles database connection, data extraction, and preprocessing.

    This class is designed to connect to a specific database (`SEG301`),
    fetch data from a hardcoded list of tables, and apply a series of
    standardized cleaning and transformation rules. The final output is a
    dictionary of clean Pandas DataFrames, ready for downstream processing
    such as embedding or analysis.

    An instance of this class maintains an active database connection until the
    `getdata` method has completed its execution.

    Attributes:
        HOST (str): The database host address.
        PASSWORD (str): The password for the database user.
        USER (str): The username for the database connection.
        connection (pymysql.Connection): The active database connection object.
        cursor (pymysql.cursors.DictCursor): The cursor object, configured to
            return rows as dictionaries.
    """
    def __init__(self,
                 HOST=os.getenv("HOST"),
                 PASSWORD=os.getenv("PASSWORD"),
                 USER=os.getenv("USER")
                 ):
        """
        Initializes the LoadData class and establishes a database connection.

        Args:
            HOST (str, optional): The database host. Defaults to the "HOST"
                environment variable.
            PASSWORD (str, optional): The database password. Defaults to the
                "PASSWORD" environment variable.
            USER (str, optional): The database user. Defaults to the "USER"
                environment variable.
        """
        self.HOST = HOST
        self.PASSWORD = PASSWORD
        self.USER = USER

        self.timeout = 10
        self.connection = pymysql.connect(
            charset="utf8mb4",
            connect_timeout=self.timeout,
            cursorclass=pymysql.cursors.DictCursor,
            db="SEG301",
            host=self.HOST,
            password=self.PASSWORD,
            read_timeout=self.timeout,
            port=22064,
            user=self.USER,
            write_timeout=self.timeout,
        )

        self.cursor = self.connection.cursor()

    @staticmethod
    def preprocessed(data):
        """
        A static utility method to remove duplicate rows based on the 'title'.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with duplicate titles removed, keeping
                the first occurrence.
        """
        data = data.drop_duplicates(subset='title', keep='first').reset_index(drop=True)
        return data

    def preprocess_data(self, data):
        """
        Applies a full cleaning and preprocessing pipeline to a DataFrame.

        The pipeline includes:
        1. Filtering out rows with specific error messages or short content.
        2. Deduplicating rows by title.
        3. Normalizing the 'content' text by removing newlines, hashtags,
           and stripping whitespace.
        4. Resetting the DataFrame index.

        Args:
            data (pd.DataFrame): The raw DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned and preprocessed DataFrame.
        """
        data = data[
            ~((data['content'].str.contains("ERROR: Failed to retrieve content.", na=False)) |
              (data['content'].str.len() < 35))
        ]
        data = self.preprocessed(data)

        data['content'] = data['content'].replace(r'[\r\n]+', ' ', regex=True).replace(r'#', '', regex=True).str.strip()

        data = data.reset_index(drop=True)
        return data

    def getdata(self):
        """
        Executes the main data loading and processing workflow.

        This method orchestrates the entire process:
        1.  Fetches data from a predefined set of tables into DataFrames.
        2.  Performs initial normalization (lowercase columns, rename 'link').
        3.  Applies the full preprocessing pipeline to each DataFrame.
        4.  Closes the database connection and cursor.
        5.  Returns the final, cleaned data.

        Returns:
            dict[str, pd.DataFrame]: A dictionary where keys are table names
                and values are the corresponding cleaned Pandas DataFrames.
        """
        queries = {
            "DiSanVanHoa": "SELECT * FROM DiSanVanHoa",
            "GiaiDoan": "SELECT * FROM GiaiDoan",
            "QuanSu": "SELECT * FROM QuanSu",
            "NhanVatLichSu": "SELECT * FROM NhanVatLichSu",
            "ThoiKy": "SELECT * FROM ThoiKy"
        }

        # ==== Push data to DataFrame ====
        dataframes = {}
        for name, query in queries.items():
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            df = pd.DataFrame(results)
            dataframes[name] = df
            print(f">>> Loaded table '{name}' with {len(df)} rows")

        for name, df in dataframes.items():
            df.columns = df.columns.str.lower()
            if 'link' in df.columns:
                df.rename(columns={'link': 'url'}, inplace=True)

        disanvanhoa_df = dataframes["DiSanVanHoa"].dropna().reset_index(drop=True)
        disanvanhoa_df = self.preprocess_data(disanvanhoa_df)

        giaidoan_df = dataframes["GiaiDoan"].dropna().reset_index(drop=True)
        giaidoan_df = self.preprocess_data(giaidoan_df)

        quansu_df = dataframes["QuanSu"].dropna().reset_index(drop=True)
        quansu_df = self.preprocess_data(quansu_df)

        nhanvatlichsu_df = dataframes["NhanVatLichSu"].dropna().reset_index(drop=True)
        nhanvatlichsu_df = self.preprocess_data(nhanvatlichsu_df)

        thoiky_df = dataframes["ThoiKy"].dropna().reset_index(drop=True)
        thoiky_df = self.preprocess_data(thoiky_df)

        self.cursor.close()
        self.connection.close()

        return {
            "DiSanVanHoa": disanvanhoa_df,
            "GiaiDoan": giaidoan_df,
            "QuanSu": quansu_df,
            "NhanVatLichSu": nhanvatlichsu_df,
            "ThoiKy": thoiky_df
        }
