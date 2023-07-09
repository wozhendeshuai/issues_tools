import spider.sql_thread as sql_thread


# Get a list of all tables in the database
tables = sql_thread.execute_select_query("SELECT table_name FROM information_schema.tables WHERE  table_schema = 'github_issues_db';")

# Initialize a dictionary to store the count of labels in different tables
label_count = {}

# Iterate over each table
for table in tables:
    table_name = table[0]
    labels = sql_thread.execute_select_query(f"SELECT labels FROM {table_name};")

    # Count the occurrence of each label in the table
    if table_name not in label_count:
        label_count[table_name] = {}

    for label in labels:
        if label[0] in label_count[table_name]:
            label_count[table_name][label[0]] += 1
        else:
            label_count[table_name][label[0]] = 1

# Print the labels and their count in different tables
for table_name, label_count in label_count.items():
    for label, count in label_count.items():
        print(f"table_name: {table_name} Label: {label}, Count: {count}")


