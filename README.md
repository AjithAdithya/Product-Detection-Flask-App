# Product Detection and Identification on Shelves

A Deep Learning Neural Network Model which runs on Faster-RCNN and trained to Detect 4 Products (Tide,Horlicks,Harpic,Kellogs) from Images of Products on Shelves.

A UI has been built with the model deployed on a server, the user can upload an Image and fins out what products are present in the Image

## Cloning the repository

Clone the git repository by running git bash in your computer and run the following command

`git clone https://github.com/AjithAdithya/Product-Detection-Flask-App.git`

Or click on the download button and extract the zip file

## Create a Virtual Environment

Run the following command in your home directory

`pip install virtualenv`

Open commandline in the repository folder and execute the following commands

`virtualenv nameofenv`

On Windows,

To activate the Virtual Environment

`nameofenv\Scripts\activate`

To deactivate the Virtual Environment and use your default environment

`nameofenv\Scripts\deactivate`

Using a Virtual Environment will help you manage custom required dependencies of any particular project/program

## Installing Dependencies

Make sure that your Virtual Environment is active, your command line will look something like

(nameofenv)C:\Users\Downloads\Product-Detection-Flask-App>

Run the following command to install all dependencies

`pip install -r requirements.txt`

## Running the Flask App
![Running the Flask App](flaskgif.gif)

On Windows,

Run the following commands while in the repository folder and in the Virtual Environment

`set FLASK_APP=app.py`

`flask run` or `python -m flask run`

Your app will be served locally and the adress will be specified. Open the same address in a browser to Use the UI.



