from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# List of URLs to scrape
urls_to_scrape = [
    "https://csds.gsu.edu/directory/?wpvcomputerscienceroles=core-faculty&wpv_aux_current_post_id=329&wpv_view_count=5249-TCPID329",
    "https://csds.gsu.edu/phdstudents/",
    "https://csds.gsu.edu/industry-advisory-board/",
    "https://csds.gsu.edu/employment-opportunities/",
    "https://cas.gsu.edu/program/computer-science-bs/",
    "https://cas.gsu.edu/program/data-science-bs/",
    "https://cas.gsu.edu/program/computer-science-bs-ms/",
    "https://csds.gsu.edu/undergraduate/#certificate-in-data-science",
    "https://csds.gsu.edu/undergraduate/#certificate-in-cybersecurity",
    "https://csds.gsu.edu/undergraduate-faqs/",
    "https://csds.gsu.edu/research/#research-areas",
    "https://csds.gsu.edu/centers-institutes/",
    "https://csds.gsu.edu/research-groups-labs/",
    "https://csds.gsu.edu/undergraduate-research/"
]

# Set up Selenium (Chrome)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Provide path to the ChromeDriver
service = Service("chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)

# Function to extract visible text
def extract_visible_text(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    visible_text = soup.get_text(separator=" ", strip=True)
    return visible_text

# Function to extract URLs
def extract_page_urls(soup):
    urls = []
    for link in soup.find_all('a', href=True):
        urls.append(link['href'])
    return urls

# Open the output file for writing
with open('gsu_webpages_data.txt', 'w', encoding='utf-8') as file:
    for url in urls_to_scrape:
        # Load each URL
        driver.get(url)
        time.sleep(5)  # Wait for the page to fully load

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract visible text and URLs
        visible_text = extract_visible_text(soup)
        page_urls = extract_page_urls(soup)

        # Write the URL to the file
        file.write(f"URL: {url}\n\n")

        # Write the visible text to the file
        file.write("Visible Text:\n")
        file.write(visible_text)
        file.write("\n\n")

        # Write the URLs to the file
        file.write("URLs:\n")
        file.write("\n".join(page_urls))
        file.write("\n\n" + "="*80 + "\n\n")  # Add a separator between pages

# Close the Selenium browser session
driver.quit()

print("Data successfully written to 'gsu_webpages_data.txt'")