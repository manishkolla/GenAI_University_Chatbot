from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin

# Set up Selenium (Chrome)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Provide path to the ChromeDriver
service = Service("chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)


# Function to extract visible text
# Function to extract visible text
def extract_visible_text(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    visible_text = soup.get_text(separator=" ", strip=True)
    return visible_text

# Function to extract URLs
def extract_page_urls(soup, base_url):
    urls = []
    for link in soup.find_all('a', href=True):
        url = link['href']
        # Convert relative URLs to absolute URLs
        full_url = urljoin(base_url, url)
        urls.append(full_url)
    return urls

# Function to scrape a webpage
def scrape_page(url, file):
    driver.get(url)
    time.sleep(5)  # Wait for the page to fully load
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Extract and write visible text
    visible_text = extract_visible_text(soup)
    file.write(f"URL: {url}\n\n")
    file.write("Visible Text:\n")
    file.write(visible_text)
    file.write("\n\n")
    
    # Extract and write URLs
    page_urls = extract_page_urls(soup, url)
    file.write("URLs:\n")
    file.write("\n".join(page_urls))
    file.write("\n\n" + "="*80 + "\n\n")
    
    return page_urls

# URL to scrape
initial_url = "https://admissions.gsu.edu/"

# Open the output file for writing
with open('gsu_admissions_data_organized.txt', 'w', encoding='utf-8') as file:
    # Scrape the admissions page
    page_urls = scrape_page(initial_url, file)
    
    # Scrape URLs found on the admissions page
    # Ensure to only visit unique URLs
    visited_urls = set([initial_url])
    urls_to_visit = set(page_urls)
    
    while urls_to_visit:
        url = urls_to_visit.pop()
        if url not in visited_urls:
            visited_urls.add(url)
            try:
                new_urls = scrape_page(url, file)
                urls_to_visit.update(new_urls)
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

# Close the Selenium browser session
driver.quit()

print("Data successfully written to 'gsu_admissions_data.txt'")