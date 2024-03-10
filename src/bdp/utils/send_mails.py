import time
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import email, smtplib, ssl

import pandas as pd
import json

smtp_server = "smtp.gmail.com"
port = 587  # For starttls
sender_email = "deep.bayesian.models@gmail.com"
password = "levyprocesses"

class SimpleMail():
    """
    Here we follow
    https://realpython.com/python-send-email/
    """
    def __init__(self,**kwargs):
        receiver_email = kwargs.get("receiver_email","ojedamarin@tu-berlin.de")
        self.receiver_email = receiver_email

    def set_headers(self,receiver_email,subject):
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message["Bcc"] = receiver_email  # Recommended for mass emails
        return message

    def set_server(self):
        # Create a secure SSL context
        context = ssl.create_default_context()
        try:
            server = smtplib.SMTP(smtp_server, port)
            server.ehlo()  # Can be omitted
            server.starttls(context=context)  # Secure the connection
            server.ehlo()  # Can be omitted
            server.login(sender_email, password)
            # TODO: Send email here
        except Exception as e:
            # Print any error messages to stdout
            print(e)
        return server

    def set_attachment(self,filenames,message):
        for filename in filenames:
            # Open PDF file in binary mode
            with open(filename, "rb") as attachment:
                # Add file as application/octet-stream
                # Email client can usually download this automatically as attachment
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header("Content-Disposition",
                            f"attachment; filename= {filename}",)

            # Add attachment to message and convert message to string
            message.attach(part)
        text = message.as_string()
        return text

    def send_mail(self,receiver_email,subject,body,filenames):
        # Add body to email
        message = self.set_headers(receiver_email,subject)
        message.attach(MIMEText(body, "plain"))
        text = self.set_attachment(filenames,message)
        server = self.set_server()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

if __name__=="__main__":
    data_from_dict = {"index": ["flows", "vae"],
                      "dl": [3, 47],
                      "rl": [18, 19]}
    results_identifier = str(int(time.time()))
    file_name1 = "results_{0}.csv".format(results_identifier)
    df = pd.DataFrame(data_from_dict)
    df.to_csv(file_name1)
    df = pd.DataFrame(data_from_dict)

    data_from_dict = {"index": ["flows", "vae"],
                      "dl": [3, 7],
                      "rl": [18, 29]}
    results_identifier = str(int(time.time())+1)
    file_name2 = "results_{0}.csv".format(results_identifier)
    df = pd.DataFrame(data_from_dict)
    df.to_csv(file_name2)
    df = pd.DataFrame(data_from_dict)
    #===========================================
    # CREATE BODY
    #===========================================
    best_results_models_folders = []
    best_results_models_folders.append(
        "/home/cesarali/projects/deep_random_fields/results/synthetic_schemata/1602861474/")
    best_results_models_folders.append(
        "/home/cesarali/projects/deep_random_fields/results/synthetic_schemata/1602938792/")

    best_results_models_folders2 = []
    for brf in best_results_models_folders:
        brf = brf.replace("/home/cesarali/", "scp -r cesarali@cluster.ml.tu-berlin.de:~/")
        brf += " ./"
        best_results_models_folders2.append(brf)

    body = "To copy the best models \n\n"
    body += "\n".join(best_results_models_folders2)

    #==============================================
    # SEND MAIL
    #==============================================
    subject = "An email with attachment from Python"
    receiver_email = "ojedamarin@tu-berlin.de"
    receiver_email = "cesarali07@gmail.com"

    filenames = [file_name1,file_name2]
    mail_object = SimpleMail()
    mail_object.send_mail(receiver_email,subject,body,filenames)
