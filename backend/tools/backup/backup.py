import sys

import pandas as pd
from sqlalchemy import create_engine, inspect
import argparse
import logging
import json
from contextlib import contextmanager
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from config.db import DBSettings


db_settings = DBSettings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
connection_string = db_settings.dsn_sync


@contextmanager
def db_connection():
    engine = create_engine(connection_string)
    try:
        yield engine
    finally:
        engine.dispose()


def csv_to_db(file_name_csv: str, table_naming: str):
    with db_connection() as engine:
        inspector = inspect(engine)
        if inspector.has_table(table_naming):
            logging.info(f"Таблица {table_naming} уже существует. Данные будут добавлены.")
        else:
            logging.info(
                f"Таблица {table_naming} не существует. Она будет создана и данные будут добавлены."
            )

        df = pd.read_csv(file_name_csv)
        # Обработка ошибок которые могут возникать если мы храним списки.
        # if "url_img_s3" in df.columns:
        #     df["url_img_s3"] = df["url_img_s3"].apply(
        #         lambda x: x.strip("[]").replace("'", "").split(", ")
        #     )

        df.to_sql(con=engine, name=table_naming, if_exists="append", index=False)

        logging.info(f"Данные успешно загружены из {file_name_csv} в таблицу {table_naming}")


def db_to_csv(file_name_csv: str, table_naming: str):
    with db_connection() as engine:
        query = f"SELECT * FROM {table_naming}"
        df = pd.read_sql(query, engine)
        df.to_csv(file_name_csv, index=False)
        logging.info(f"Данные успешно выгружены из таблицы {table_naming} в CSV {file_name_csv}")


def bulk_csv_to_db(config_path: str):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        for file_name_csv, table_naming in config.items():
            csv_to_db(file_name_csv, table_naming)


def export_all_tables(output_dir: str):
    with db_connection() as engine:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        os.makedirs(output_dir, exist_ok=True)
        for table in tables:
            file_name_csv = os.path.join(output_dir, f"{table}.csv")
            db_to_csv(file_name_csv, table)
        logging.info(f"Все таблицы успешно выгружены в директорию {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Перенос данных между CSV и PostgreSQL")
    parser.add_argument(
        "command",
        choices=[
            "csv_to_db",
            "db_to_csv",
            "bulk_csv_to_db",
            "export_all",
            "export_selected",
        ],
        help="Команда для выполнения",
    )
    parser.add_argument("--file_name_csv", type=str, help="Имя файла CSV")
    parser.add_argument("--table_naming", type=str, help="Имя таблицы базы данных")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Путь к файлу конфигурации для массовой загрузки CSV",
    )
    parser.add_argument("--output_dir", type=str, help="Директория для выгрузки CSV файлов")
    parser.add_argument("--tables", nargs="+", help="Список таблиц для выгрузки")

    args = parser.parse_args()

    if args.command == "csv_to_db":
        csv_to_db(args.file_name_csv, args.table_naming)
    elif args.command == "db_to_csv":
        db_to_csv(args.file_name_csv, args.table_naming)
    elif args.command == "bulk_csv_to_db":
        bulk_csv_to_db(args.config_path)
    elif args.command == "export_all":
        export_all_tables(args.output_dir)
    else:
        raise


if __name__ == "__main__":
    main()
