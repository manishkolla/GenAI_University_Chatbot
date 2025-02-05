#Importing libraries for scraper, please ensure you have the respective drivers installed and path
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
chrome_options.add_argument("--incognito")  # Open browser in incognito mode
service = Service('chromedriver.exe')  # Specify the ChromeDriver executable

# Initialize the Chrome driver with specified options
driver = webdriver.Chrome(service=service, options=chrome_options)

# Define the website URL to scrape
website = 'https://cas.gsu.edu/profile-directory/'
driver.get(website)  # Open the webpage

# Lists to store extracted details
details = []
professors = []

# Loop through 21 pages of the directory
for i in range(1, 22):
    url = f"https://cas.gsu.edu/profile-directory/?wpv_aux_current_post_id=13610&wpv_view_count=13592-TCPID13610&wpv_paged={i}"
    print(url)  # Print the current URL being scraped
    driver.get(url)  # Load the page
    time.sleep(5)  # Wait for page elements to load

    # Find all profile rows on the page
    names = driver.find_elements(By.ID, "profile_row")
    
    for x in names:
        element = x.find_elements(By.CLASS_NAME, "vc_column_container ")  # Extract professor details
        professor = []
        
        for x in element: 
            try:
                professor.append(x.text)  # Append extracted text to the professor list
                if professor.count('') > 3:  # Skip if too many empty values
                    continue
            except:
                pass  # Ignore exceptions and continue execution
        
        professors.append(professor)  # Add professor details to the main list

# Open a text file in write mode to save extracted data
with open(r"C:\Users\mkolla1\OneDrive - Georgia State University\Desktop\Fall 2024\AI\Project\professors_directory-2.txt", "w", encoding="utf-8") as file:
    for o in professors:
        # Join the elements in each professor list with a comma and write to the file
        file.write(", ".join(o) + "\n")
