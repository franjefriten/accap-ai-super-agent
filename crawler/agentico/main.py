from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy, DFSDeepCrawlStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter
)

from pathlib import Path

from aiofiles import open as aio_open
import aiohttp
import asyncio
import os
import json
import logging
import logging.config
import re

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, WebDriverException
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import time
from datetime import datetime

import pandas as pd
import numpy as np
from docling.document_converter import DocumentConverter

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from url_to_llm_text.get_html_text import get_page_source   # you can also use your own code or other services to get the page source
from url_to_llm_text.get_llm_input_text import get_processed_text   # pass html source text to get llm ready text

from openai import AzureOpenAI

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# cargamos configuacion de logging
logging.config.fileConfig('crawler/agentico/logconfig.conf')

# creamos logger
logger = logging.getLogger(__name__)

# cargamos variables de entorno
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv(".env")


class CallSchema(BaseModel):
    #title: str = Field(..., validation_alias="nombre", description="Title of the AEI entry")
    start_date: str = Field(..., validation_alias="fecha_inicio", description="Fecha de inicio del período de aplicabilidad de la convocatoria")
    end_date: str = Field(..., validation_alias="fecha_final", description="Fecha de finalización del período de aplicabilidad de la convocatoria")
    finalidad: str = Field(..., validation_alias="descripcion", description="Finalidad de la convocatoria")
    amount: str = Field(..., validation_alias="presupuesto", description="Presupuesto de la convocatoria")
    type_of_subsidy: str = Field(..., validation_alias="tipo", description="Tipo de subvención de la convocatoria")
    beneficiary: str = Field(..., validation_alias="beneficiario", description="Beneficiarios de la convocatoria")
    place: str = Field(..., validation_alias="localidad", description="Localidad de la convocatoria")
    #url: str = Field(..., description="URL of the AEI entry")
    bases: str = Field(..., validation_alias="bases", description="URL a las bases de la orden de la convocatoria")
    compatibility: str = Field(..., validation_alias="compatibilidad", description="Compatibilidad de la convocatoria")
    duration: str = Field(..., validation_alias="duracion", description="Duración de la convocatoria")


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', LegacySSLAdapter())


def check_link_behavior(url):
    """
    Determines what happens when accessing a URL:
    - 'download': Forces PDF download
    - 'view': Opens PDF in browser
    - 'redirect': Redirects to another URL 
    - 'error': Error occurred
    - 'unknown': Not a PDF
    """
    session = requests.Session()
    session.mount('https://', HTTPAdapter())
    
    try:
        # First try HEAD request (doesn't download content)
        response = session.head(url, allow_redirects=True, timeout=5)
        
        # Check for direct PDF responses
        if 'application/pdf' in response.headers.get('Content-Type', ''):
            if 'attachment' in response.headers.get('Content-Disposition', ''):
                return 'download'
            else:
                return 'view'
        
        # Check for redirects
        if response.url != url:
            return ('redirect', response.url)
        
        # Fallback to GET if HEAD fails (some servers block HEAD)
        response = session.get(url, stream=True, timeout=5)
        
        if 'application/pdf' in response.headers.get('Content-Type', ''):
            if 'attachment' in response.headers.get('Content-Disposition', ''):
                return 'download'
            return 'view'
        
        return 'unknown'
    
    except Exception as e:
        return ('error', str(e))


def download_pdf(url, driver, options=None):
    tipo = check_link_behavior(url)
    if tipo == "download":
        driver.get(url)
    elif tipo == "view":
        # Suponemos configuración del navegador para descargar PDFs automáticamente
        # de tal forma que no se abra el visor de PDF
        driver.get(url)
    elif tipo[0] == "redirect":
        # Vamos a suponer solo el primer pdf
        pdf = driver.find_element(By.XPATH, "//a[contains(@href, 'pdf') or contains(@href, 'documento')]")
        download_pdf(pdf.get_attribute('href'), driver)

async def _CrawlAEI():
    
    LOG_FILE = "crawler/logger.txt"
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    URLS = URLS["AEI"][:-1]
    
    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=[
            r"^https://www\.aei\.gob\.es/convocatorias/buscador\-convocatorias/[\w\-]+",
        ]),

        # Only crawl specific domains
        # DomainFilter(
            # allowed_domains=["https://www.pap.hacienda.gob.es"],
        # ),

        # Only include specific content types
        # ContentTypeFilter(allowed_types=["html"])
    ])
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=2,
        include_external=False,
        logger=logger,
        filter_chain=filter_chain
    )
    extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="azure/gpt-4o-mini",
            api_token=os.getenv("AZURE_AI_AGENT_API_KEY"),
            base_url=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        ),
        instruction="""Extract the following information from the webpage: \n\n
                    1. Title of the AEI entry\n
                    2. Term of the AEI entry\n
                    3. Description of the AEI entry\n
                    4. Amount of the AEI entry\n
                    5. Type of the AEI entry\n
                    6. Beneficiary of the AEI entry\n
                    7. Entity of the AEI entry\n
                    8. URL of the AEI entry\n
                    9. URL to the order basis of the AEI entry\n\n
                    Please provide the information in JSON format""",
        schema=CallSchema.model_json_schema(),
        extraction_type="json",
        input_format="html"
    )
    browser_config = BrowserConfig(
        verbose=True,
        accept_downloads=True,
        java_script_enabled=True,
        ignore_https_errors=True
    )
    contenido = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        #if not isinstance(convocatorias, list): # a instancia es una url
        load_more_js = [
            "window.scrollTo(0, document.body.scrollHeight);",
            # The "More" link at page bottom
            "document.querySelector('a.page-link')?.click();"  
        ]
        session_id = "hn_session"
        for i in range(3):
            run_config = CrawlerRunConfig(
                deep_crawl_strategy=deep_crawl_strategy,
                scraping_strategy=LXMLWebScrapingStrategy(logger=logger),
                extraction_strategy=extraction_strategy,
                session_id=session_id,
                wait_for="""js:() => {
                    return document.readyState === 'complete'
                }""",
                wait_until="load",
                page_timeout=10000
            )
            result = await crawler.arun(url=URLS, config=run_config)
            contenido.append([{**json.loads(res.extracted_content)[0], "url": os.path.join(URLS, res.url)} for res in result[1:] if res.extracted_content is not None])
            if i == 3: #not len(result) > 1:
                logger.info("Busqueda finalizada")
                break
            logger.info("Cargando siguiente pagina")
            #await write_log(LOG_FILE, result, regex=r"^https://www.ciencia.gob.es/Convocatorias/*")
            next_page_conf = CrawlerRunConfig(
                js_code=load_more_js,
                wait_for="""js:() => {
                    return document.querySelectorAll('div.item-list ul li').length > 10;
                }""",
                # Mark that we do not re-navigate, but run JS in the same session:
                js_only=True,
                session_id=session_id
            )
            # Re-use the same crawler session
            result2 = await crawler.arun(
                url=URLS,  # same URL but continuing session
                config=next_page_conf
            )

    return contenido


async def AgenticoAEI():

    LOG_FILE = "crawler/logger.txt"
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    url = URLS["AEI"]
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    download_pdfs = True
    # Problemas con los certificados de seguridad
    # certificados anticuados

    client = AzureOpenAI(  
        azure_endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_AGENT_API_KEY"),
        api_version="2025-01-01-preview",
    )

    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": 
                        """
                        You are a web scraping assistant. Your task is to extract specific information from web pages.
                        This is the information you need to extract:
                        """ + str(CallSchema.model_json_schema()) + """
                        Please provide the information in JSON format and leave any unfound data as '' in the entry.
                        Here is the text:
                        """
                }
            ]
        }
    ]

    contenido = []

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--ignore-certificate-errors")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    prefs = {
        "plugins.always_open_pdf_externally": True,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True
    }
    options.add_experimental_option(
        'prefs', prefs)


    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)
    driver.get(url=url)

    for page in range(3):
        
        wait.until(lambda driver: len(driver.find_elements(By.XPATH, '//div[contains(@class, "item-list")]//li')) >= 5)

        # Get all call links on current page
        call_elements = driver.find_elements(By.XPATH, '//div[contains(@class, "item-list")]//li//a[contains(@href, "convocatorias")]')
        call_urls = [elem.get_attribute('href') for elem in call_elements]

        for call_url in call_urls:
            print(f"Scraping: {call_url}")

            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(call_url)

            try:
                # Wait for main content to load
                wait.until(EC.presence_of_element_located((By.XPATH, '//div[@role="main"]')))
                
                nombre = driver.find_elements(By.XPATH, '//h1')[1].text


                table = driver.find_element(By.XPATH, '//table[contains(@class, "table-striped")]')
                page_source = table.get_attribute('innerHTML')
                llm_text = await get_processed_text(page_source, call_url)
                chat_prompt[0]['content'][0]['text'] += llm_text
                completion = client.chat.completions.create(  
                    model="gpt-4o-mini",
                    messages=chat_prompt,
                    max_tokens=800,  
                    temperature=0.7,  
                    top_p=0.95,  
                    frequency_penalty=0,  
                    presence_penalty=0,
                    stop=None,
                    stream=False  
                )
                call_data = completion.choices[0].message.content
                call_data = re.sub(r"```json\s*|```", "", (call_data).strip())
                call_data = dict(json.loads(call_data))
                
                call_data["nombre"] = nombre  
                call_data["url"] = call_url
                        
                try:
                    docs_section = driver.find_element(By.XPATH, '//aside[@role="complementary"]//ul')
                    documents = docs_section.find_elements(By.TAG_NAME, 'a')
                    for doc in documents:
                        if 'bases' in doc.text.lower():
                            call_data['bases'] = doc.get_attribute('href')
                #         if 'convocatoria' in doc.text.lower() and download_pdfs == True:
                #             convocatoria_url = doc.get_attribute('href')
                #             try:
                #                 driver.execute_script("window.open('');")
                #                 driver.switch_to.window(driver.window_handles[2])
                #                 driver.get(convocatoria_url)
                #                 store_folder = os.path.join("./downloads", call_url.split("/")[-1])
                #                 if not os.path.isdir(store_folder):
                #                     os.makedirs(store_folder)
                #                 prefs["download.default_directory"] = store_folder
                #                 options.add_experimental_option('prefs', prefs)
                #                 pdfs_convocatoria = driver.find_elements(By.XPATH, '//a[@type="application/pdf"]')
                #                 for pdf in pdfs_convocatoria:
                #                     pdf.click()
                #                     # pdf_url = pdf.get_attribute('href')
                #                     # pdf_name = pdf_url.split("/")[-1]
                #                     # pdf_path = os.path.join(store_folder, pdf_name)
                #                     # if not os.path.isfile(pdf_path):
                #                     #     with session.get(pdf_url, stream=True, timeout=10) as r:
                #                     #         r.raise_for_status()
                #                     #         with open(pdf_path, 'wb') as f:
                #                     #             for chunk in r.iter_content(chunk_size=8192):
                #                     #                 f.write(chunk)
                #             except Exception as e:
                #                 print(f"Error opening convocatoria URL: {str(e)}")
                    contenido.append(call_data)
                except NoSuchElementException:
                    call_data['bases'] = None
                    contenido.append(call_data)

            except Exception as e:
                print(f"Error scraping {call_url}: {str(e)}")
            finally:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(2)  # Be polite
        # Try to go to next page
        try:
            next_btn = driver.find_element(By.XPATH, '//a[contains(@class, "page-link") and contains(text(), "Siguiente")]')
            if 'disabled' in next_btn.get_attribute('class'):
                break
            next_btn.click()
            time.sleep(3)  # Wait for page load
        except NoSuchElementException:
            break

    driver.quit()
    return contenido


async def AgenticoSNPSAP():
    pdf_filename = f"listado{datetime.today().month}_{datetime.today().day}_{datetime.today().year}.pdf" 
    LOG_FILE = "crawler/logger.txt"
    
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    base_url = URLS["SNPSAP"]

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_AGENT_API_KEY"),
        api_version="2025-01-01-preview",
    )
    chat_prompt_scrap_web = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": 
                        """
                        You are a web scraping assistant. Your task is to extract specific information from web pages.
                        This is the information you need to extract:
                        {schema}
                        Please provide the information in JSON format, in case of a field missing, leave an empty string. Here is the text:
                        """
                }
            ]
        }
    ]
    chat_prompt_scrap_pdf = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": 
                        """
                        You are a pdf scraping assistant.
                        We have information already extracted from another source, but you need to complete it.
                        This is the information we have:
                        {data_extracted}
                        This is the information you need to extract:
                        {schema}
                        Please provide the information in JSON format, in case of a field missing, leave an empty string. Here is the text:
                        {text}
                        """
                }
            ]
        }
    ]
        
    # Configure Firefox options for automatic PDF downloading
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-features=DownloadBubble")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    prefs = {
        "plugins.always_open_pdf_externally": True,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.plugins_disabled": ["Chrome PDF Viewer"],
        "pdfjs.disabled": True  # For older Chrome versions
    }
    options.add_experimental_option(
        'prefs', prefs)
    

    if pdf_filename not in os.listdir("./downloads/pdf"):
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 20)
        prefs["download.default_directory"] = "./downloads/pdf"
        options.add_experimental_option('prefs', prefs)
        driver.get(base_url)
        # Wait until a PDF link appears (Modify XPath if necessary)
        try:
            pdf_link = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//a[@class='mat-focus-indicator mat-icon-button mat-button-base']"))
            )

            # Click the link to start the download
            pdf_link.click()

            print("PDF download initiated successfully!")

            # Wait a few seconds for the file to download
            time.sleep(10)
            driver.quit()

        except Exception as e:
            print("Error: PDF link not found.", str(e))
            driver.quit()
    
    input_doc_path = Path(os.path.join('downloads/pdf', pdf_filename))
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)
    df = pd.DataFrame()
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        df = pd.concat([df, table_df], ignore_index=True, axis=0)
        logging.info(f"Data extraída de tabla {table_ix}")
    df["url"] = list(map(lambda x: os.path.join(base_url, x), df["Código BDNS"]))
    df = df[["Código BDNS", "Órgano", "Fecha de registro", "Título", "url"]]
    df[[
        "presupuesto",
        "fecha_inicio",
        "fecha_final",
        "finalidad",
        "localidad",
        "tipo",
        "bases",
        "beneficiario", 
        "compatibilidad",
        "duracion"
    ]] = pd.DataFrame(
        data=[],
        columns=[
            "presupuesto",
            "fecha_inicio",
            "fecha_final",
            "finalidad",
            "localidad",
            "tipo",
            "bases",
            "beneficiario",
            "compatibilidad",
            "duracion"
        ],
        dtype=str
    )
    for i, url in enumerate(df["url"]):
        try:
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 20)
            driver.get(url)
            text = driver.find_element(By.TAG_NAME, 'main').get_attribute('innerHTML')
            text = await get_processed_text(text, url)
            chat_prompt_scrap_web[0]['content'][0]['text'] = \
                chat_prompt_scrap_web[0]['content'][0]['text'].format(
                    schema=CallSchema.model_json_schema(),
                )
            chat_prompt_scrap_web[0]['content'][0]['text'] += text
            completion = client.chat.completions.create(  
                model="gpt-4o-mini",
                messages=chat_prompt_scrap_web,
                max_tokens=800,  
                temperature=0.7,  
                top_p=0.95,  
                frequency_penalty=0,  
                presence_penalty=0,
                stop=None,
                stream=False  
            )
            table_data = completion.choices[0].message.content
            table_data = re.sub(r"```json\s*|```", "", (table_data).strip())
            table_data = dict(json.loads(table_data))
        except Exception as e:
            print(f"Error al obtener los datos de las tablas: {e}")
        finally:
            driver.quit()
        try:
            bases = table_data["bases"]
            store_folder = os.path.join("./downloads", df.loc[i, "Código BDNS"])
            if not os.path.isdir(store_folder):
                os.makedirs(store_folder)
            prefs["download.default_directory"] = os.path.abspath(store_folder)
            options.add_experimental_option('prefs', prefs)           
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 20)
            download_pdf(bases, driver, options)
            pdf_data = {}
            for file in os.listdir(store_folder):
                if not file.endswith(".pdf"):
                    continue
                reader = PdfReader(os.path.join(store_folder, file))
                text = "\n".join([page.extract_text() for page in reader.pages])
                chat_prompt_scrap_pdf_text = chat_prompt_scrap_pdf[0]['content'][0]['text'].format(
                        schema=CallSchema.model_json_schema(),
                        data_extracted=str(table_data),
                        text=text
                    )
                chat_prompt_scrap_pdf[0]['content'][0]['text'] = chat_prompt_scrap_pdf_text
                completion = client.chat.completions.create(  
                    model="gpt-4o-mini",
                    messages=chat_prompt_scrap_pdf,
                    max_tokens=800,  
                    temperature=0.7,  
                    top_p=0.95,  
                    frequency_penalty=0,  
                    presence_penalty=0,
                    stop=None,
                    stream=False  
                )
                data = completion.choices[0].message.content
                pdf_data_i = re.sub(r"```json\s*|```", "", (data).strip())
                pdf_data_i = dict(json.loads(pdf_data_i))
                pdf_data = {**pdf_data, **pdf_data_i}                    
        except (WebDriverException, NoSuchElementException) as e:
            print(f"Error al descargar el PDF: {e}")
        except PdfReadError as e:
            print(f"Error al leer el PDF: {e}")
        finally:
            driver.quit()
        call_data = {**table_data, **pdf_data}
        df.loc[i, call_data.keys()] = call_data.values()
    
    return df


if __name__ == "__main__":
    asyncio.run(AgenticoSNPSAP())
    #print(json.dumps(AEISchema.model_json_schema()["properties"], indent=4))