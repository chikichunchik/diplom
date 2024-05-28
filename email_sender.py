import datetime
import smtplib
from email.message import EmailMessage
from sqlalchemy import create_engine
import pandas as pd



class EmailAlerts():
    def __init__(self):
        DATABASE_URL = f"postgresql://postgres:admin@localhost:5433/postgres"
        self.host = "imap.gmail.com"
        self.username = "andrew_h@dataforest.ai"
        self.password = 'bstahvshemichpjb'
        self.connection = smtplib.SMTP_SSL(self.host)
        self.connection.login(self.username, self.password)
        self.engine = create_engine(DATABASE_URL)

    def send_message(self, text, file_path, email):
        msg = EmailMessage()
        msg['Subject'] = 'Daily rent alerts'
        msg['From'] = 'Andrii Honcharenko'
        msg['To'] = email
        msg.set_content(text)

        with open(file_path, 'rb') as f:
            file_data = f.read()
        msg.add_attachment(file_data, maintype="application", subtype="xlsx", filename=file_path)
        self.connection.send_message(msg)

    def send_alerts(self):
        df = pd.read_sql("""SELECT * FROM subscriptions""", self.engine)
        current_date = datetime.date.today()
        for index, row in df.iterrows():
            temp_df = pd.read_sql(row['search'], self.engine)
            temp_df.to_excel(f"rent_data_{current_date}.xlsx")
            self.send_message(f"Alert on {current_date}:", f"rent_data_{current_date}.xlsx",
                              row['email'])

        self.connection.quit()


if __name__ == '__main__':
    test = EmailAlerts()
    test.send_alerts()