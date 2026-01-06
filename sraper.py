from selenium import webdriver
from selenium.webdriver.chrome.service import Service

service = Service(executable_path='path/to/chromedriver')
driver = webdriver.Chrome(service=service)



