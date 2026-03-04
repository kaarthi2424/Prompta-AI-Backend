from sqlmodel import create_engine, Session
from urllib.parse import quote_plus

server = "DESKTOP-9467S9F\\GEEK"
database = "ChatDB"
username = "sa"
password = "123"
driver = "ODBC Driver 17 for SQL Server"

params = quote_plus(
    f"DRIVER={{{driver}}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "TrustServerCertificate=yes;"
)

DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"

engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session