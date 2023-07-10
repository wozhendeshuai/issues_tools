import spider.sql_thread as sql_thread


def get_label_set():
    # Initialize a dictionary to store the count of labels in different tables
    # Get a list of all tables in the database
    tables = sql_thread.execute_select_query(
        "SELECT table_name FROM information_schema.tables WHERE  table_schema = 'github_issues_db';")
    # Initialize a dictionary to store the count of labels in different tables
    label_count = {}
    label_set = set()
    # Iterate over each table
    for table in tables:
        table_name = table[0]
        labels = sql_thread.execute_select_query(f"SELECT labels FROM {table_name};")

        # Count the occurrence of each label in the table
        if table_name not in label_count:
            label_count[table_name] = {}

        for label_str in labels:
            label_list = eval(label_str[0])
            for label in label_list:
                label_set.add(label)
                if label in label_count[table_name]:
                    label_count[table_name][label] += 1
                else:
                    label_count[table_name][label] = 1
    print(label_set)
    return label_set
