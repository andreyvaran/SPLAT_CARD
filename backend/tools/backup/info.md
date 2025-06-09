### 1.Импорт данных из CSV в базу данных

`python backup.py csv_to_db --file_name_csv path/to/your_file.csv --table_naming your_table_name
`
### 2.Экспорт данных из базы данных в CSV

`python backup.py db_to_csv --file_name_csv path/to/your_file.csv --table_naming your_table_name
`
### 3.Массовый импорт данных из CSV в базу данных (используя конфигурационный файл)

`python backup.py bulk_csv_to_db --config_path path/to/your_config.json
`
### 4.Экспорт всех таблиц из базы данных в CSV файлы

`python backup.py export_all --output_dir path/to/output_directory
`
### 5.Экспорт выбранных таблиц из базы данных в CSV файлы

`python backup.py export_selected --tables table1 table2 --output_dir path/to/output_directory
`