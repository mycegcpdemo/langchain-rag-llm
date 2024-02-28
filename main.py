from helper_classes.database import Database
from helper_classes.database_operations import DatabaseOperations

# Press the green button in the gutter to run the script.

database = Database()
db_ops = DatabaseOperations()

rs = db_ops.install_pgvector(database.get_engine())
print(rs)

