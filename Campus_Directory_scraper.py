from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--incognito") 
service = Service('chromedriver.exe')

# Initialize the Chrome driver with options
driver = webdriver.Chrome(service=service, options=chrome_options)
website='https://cas.gsu.edu/profile-directory/'
driver.get(website)
details=[]
professors=[]
for i in range(1,22):
    url=f"https://cas.gsu.edu/profile-directory/?wpv_aux_current_post_id=13610&wpv_view_count=13592-TCPID13610&wpv_paged={i}"
    print(url)
    driver.get(url)
    time.sleep(5)
    
    names= driver.find_elements(By.ID, "profile_row")
    for x in names:
        element= x.find_elements(By.CLASS_NAME, "vc_column_container ")
        professor=[]
        for x in element: 
            try:
                professor.append(x.text)
                if professor.count('') > 3:
                    continue
            except:
                pass
        professors.append(professor)

# Open a text file in write mode
with open(r"C:\Users\mkolla1\OneDrive - Georgia State University\Desktop\Fall 2024\AI\Project\professors_directory-2.txt", "w", encoding="utf-8") as file:
    for o in professors:
        # Join the elements in each professor list with a comma, then write to file
        file.write(", ".join(o) + "\n")
