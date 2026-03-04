import pyodbc
print("Installed ODBC Drivers:")
for driver in pyodbc.drivers():
    print(f" - {driver}")