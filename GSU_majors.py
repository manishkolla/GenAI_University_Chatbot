# Code to fetch all the majors and degrees from the GSU website and save them to a text file.
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd

website = 'https://www.gsu.edu/program-cards/'
service = Service('chromedriver.exe')
driver = webdriver.Chrome(service=service)
driver.get(website)


majors_list=[]
#Only majors
# majors=driver.find_elements(By.CLASS_NAME, 'program-title')
# for x in majors:
#     majors_list.append(x.text)
degree_list=[]
#Only Degrees
# degree=driver.find_elements(By.CLASS_NAME, 'degree-type')
# for y in degree:
#     degree_list.append(y.text)


#Pages itegration
for i in range(1,10):
    pg=driver.find_elements(By.CLASS_NAME, "wpv-filter-pagination-link")[i].get_attribute("href")
    driver.get(pg)
    majors=driver.find_elements(By.CLASS_NAME, 'program-title')
    for x in majors:
        majors_list.append(x.text)
    degree=driver.find_elements(By.CLASS_NAME, 'degree-type')
    for y in degree:
        degree_list.append(y.text)
# print(majors_list, degree_list)
# dictio={"Majors", majors_list}
df=pd.DataFrame({"Majors": majors_list, "Degrees": degree_list})
print(df)
df.to_csv(r"data/majors.txt", sep='\t', index=False)

