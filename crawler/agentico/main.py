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
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

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
    objective: str = Field(..., validation_alias="objetivo", description="Objetivo de la convocatoria")


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', LegacySSLAdapter())


def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result


def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def safe_json_loads(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Get the position where parsing failed
        error_pos = e.pos
        
        # Find the last unclosed quote before the error position
        last_quote_pos = json_str.rfind('"', 0, error_pos)
        
        # Check if we're inside an unclosed string
        if last_quote_pos > 0 and json_str[last_quote_pos-1] != '\\':  # Not an escaped quote
            # Insert closing quote only at the exact unterminated string
            repaired = json_str[:error_pos] + '"' + json_str[error_pos:]
        else:
            # Handle other types of truncation (objects/arrays)
            repaired = json_str[:error_pos]
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            repaired += '}' * open_braces
            repaired += ']' * open_brackets
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Final fallback - return None or raise custom error
            return {"error": "Could not repair JSON", "truncated_data": json_str[:200] + "..."}

def check_link_behavior(url: str) -> str:
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
        if url.endswith("pdf"):
            return 'view'

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
        pass
        # Vamos a suponer solo el primer pdf
        #pdf = driver.find_element(By.XPATH, "//a[contains(@href, 'pdf') or contains(@href, 'documento')]")
        #download_pdf(pdf.get_attribute('href'), driver)

def convert_table_to_json(table_data):
    """
    Transform Document Intelligence table structure into column-value JSON
    
    Input: 
        [{'kind': 'columnHeader', 'rowIndex': 0, 'columnIndex': 0, 'content': 'Año'},
         {'rowIndex': 1, 'columnIndex': 0, 'content': '2023'}]
         
    Output:
        [{'Año': '2023'}]
    """
    # Step 1: Extract headers
    print(table_data)
    headers = {}
    max_col = max(cell['columnIndex'] for cell in table_data)
    
    for cell in table_data:
        if cell.get('kind') == 'columnHeader':
            headers[cell['columnIndex']] = cell['content']
    
    # Step 2: Group by rows
    rows = {}
    for cell in table_data:
        if cell.get('kind') != 'columnHeader':  # Skip headers
            row_idx = cell['rowIndex']
            col_idx = cell['columnIndex']
            
            if row_idx not in rows:
                rows[row_idx] = {col_idx: cell['content']}
            else:
                rows[row_idx][col_idx] = cell['content']
    
    # Step 3: Map to header names
    result = []
    for row in rows.values():
        json_row = {}
        for col_idx, content in row.items():
            if col_idx in headers:  # Only include columns with headers
                json_row[headers[col_idx]] = content.strip()
        if json_row:  # Skip empty rows
            result.append(json_row)
    
    return result


def analyze_layout(pdf):
    endpoint = os.getenv("AZURE_DOCUMENT_ANALYZER_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_ANALYZER_API_KEY")

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    MAX_BYTES = 4 * 1024 * 1024

    with open(pdf, "rb") as doc:
        # Read exactly MAX_BYTES (truncates if larger)
        file_content = doc.read(MAX_BYTES)
        
        # Verify if we hit the limit
        if len(file_content) == MAX_BYTES:
            remaining = doc.read(1)  # Check for any remaining bytes
            if remaining:
                raise ValueError(f"File exceeds {MAX_BYTES} bytes limit")

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", 
            AnalyzeDocumentRequest(
                bytes_source=file_content,
            )
        )

    result: AnalyzeResult = poller.result()
    if result.tables:
        table_data = []
        for table in result.tables: 
            table_data.append(convert_table_to_json(table['cells']))
    else:
        table_data = []
    if result.key_value_pairs:
        kv_data = [kv_data.append(kv.as_dict()) for kv in result.key_value_pairs]
    else:
        kv_data = []
    if result.content:
        content = result.content
    else:
        content = ""
        #for table_idx, table in enumerate(result.tables):
        #     print(
        #         f"Table # {table_idx} has {table.row_count} rows and "
        #         f"{table.column_count} columns"
        #     )
        #     if table.bounding_regions:
        #         for region in table.bounding_regions:
        #             print(
        #                 f"Table # {table_idx} location on page: {region.page_number} is {region.polygon}"
        #             )
            # for cell in table.cells:
            #     print(
            #         f"...Cell[{cell.row_index}][{cell.column_index}] has text '{cell.content}'"
            #     )
            #     if cell.bounding_regions:
            #         for region in cell.bounding_regions:
            #             print(
            #                 f"...content on page {region.page_number} is within bounding polygon '{region.polygon}'"
            #             )
    
    return ([*kv_data, *table_data], content)

    # if result.styles and any([style.is_handwritten for style in result.styles]):
    #     print("Document contains handwritten content")
    # else:
    #     print("Document does not contain handwritten content")

    # for page in result.pages:
    #     # print(f"----Analyzing layout from page #{page.page_number}----")
    #     # print(
    #     #     f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
    #     # )

    #     if page.lines:
    #         for line_idx, line in enumerate(page.lines):
    #             words = get_words(page, line)
    #             print(
    #                 f"...Line # {line_idx} has word count {len(words)} and text '{line.content}' "
    #                 f"within bounding polygon '{line.polygon}'"
    #             )

    #             for word in words:
    #                 print(
    #                     f"......Word '{word.content}' has a confidence of {word.confidence}"
    #                 )

    #     if page.selection_marks:
    #         for selection_mark in page.selection_marks:
    #             print(
    #                 f"Selection mark is '{selection_mark.state}' within bounding polygon "
    #                 f"'{selection_mark.polygon}' and has a confidence of {selection_mark.confidence}"
    #             )

    # if result.tables:
    #     for table_idx, table in enumerate(result.tables):
    #         print(
    #             f"Table # {table_idx} has {table.row_count} rows and "
    #             f"{table.column_count} columns"
    #         )
    #         if table.bounding_regions:
    #             for region in table.bounding_regions:
    #                 print(
    #                     f"Table # {table_idx} location on page: {region.page_number} is {region.polygon}"
    #                 )
    #         for cell in table.cells:
    #             print(
    #                 f"...Cell[{cell.row_index}][{cell.column_index}] has text '{cell.content}'"
    #             )
    #             if cell.bounding_regions:
    #                 for region in cell.bounding_regions:
    #                     print(
    #                         f"...content on page {region.page_number} is within bounding polygon '{region.polygon}'"
    #                     )

    # print("----------------------------------------")


def query_pdf_data(document_tables, document_data, information_avaialble):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_AGENT_API_KEY"),
        api_version="2025-01-01-preview",
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use "gpt-35-turbo" for faster/cheaper
        messages=[
            {"role": "system", "content": """
             Eres un asistente de extracción de datos.
             Tu tarea es extraer información específica de documentos JSON.
             Sin embargo, ya tenemos parte de la información extraída,
             por lo que solo debes extraer la información faltante.
             Rellena los campos vacíos (marcados con "") con la información que encuentres en el documento.
             """},
            {"role": "user", "content": f"""
             TABLAS DEL DOCUMENTO:
             {document_tables}

             DATOS DEL DOCUMENTO:
             {document_data}
             
             INFORMACIÓN DISPONIBLE:
             {information_avaialble}

             Devuelve toda la información en formato JSON y deja los datos no encontrados como '' en la entrada.
             """}
        ],
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,
        stream=False 
    )
    
    return response.choices[0].message.content


def extract_data_from_page(text, schema):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_AGENT_API_KEY"),
        api_version="2025-01-01-preview",
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use "gpt-35-turbo" for faster/cheaper
        messages=[
            {"role": "system", "content": """
             Eres un asistente de web scraping. Tu tarea es extraer
             información específica de las páginas web.
            Esta es la información que necesitas extraer:
             """},
            {"role": "user", "content": f"""
             ESQUEMA DE DATOS A EXTRAER:
             {schema}
             
             TEXTO DE LA PÁGINA:
             {text}

             Devuelve la información en formato JSON y deja los datos no encontrados como '' en la entrada.
             """}
        ],
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,
        stream=False 
    )
    return response.choices[0].message.content

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
                        Eres un asistente de web scraping. Tu tarea es extraer información específica de las páginas web.
                        Esta es la información que necesitas extraer:
                        """ + str(CallSchema.model_json_schema()) + """
                        Proporcione la información en formato JSON y deje los datos no encontrados como '' en la entrada.
                        Aquí está el texto:
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
    pdf_filename = "listado5_6_2025.pdf" #f"listado{datetime.today().month}_{datetime.today().day}_{datetime.today().year}.pdf" 
    LOG_FILE = "crawler/logger.txt"
    
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    base_url = URLS["SNPSAP"]

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_AGENT_API_KEY"),
        api_version="2025-01-01-preview",
    )
        
    # Configure Firefox options for automatic PDF downloading
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-features=DownloadBubble")
    options.add_argument("--disable-extensions")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    prefs = {
        "plugins.always_open_pdf_externally": True,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.plugins_disabled": ["Chrome PDF Viewer"],
        "pdfjs.disabled": True  # For older Chrome versions
    }
    

    if pdf_filename not in os.listdir("./downloads/pdf"):
        prefs["download.default_directory"] = os.path.abspath("./downloads/pdf")
        options.add_experimental_option('prefs', prefs)
        driver = webdriver.Chrome(options=options)
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
    df = df[["Código BDNS", "Departamento", "Fecha de registro", "Título", "url"]]
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
        "duracion",
        "objetivo"
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
            "duracion",
            "objetivo"
        ],
        dtype=str
    )
    for i, url in enumerate(df["url"]):
        try:
            store_folder = os.path.join("./downloads", df.loc[i, "Código BDNS"])
            if not os.path.isdir(store_folder):
                os.makedirs(store_folder)
            prefs["download.default_directory"] = os.path.abspath(store_folder)
            options.add_experimental_option('prefs', prefs)           
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 20)
            driver.get(url)
            text = driver.find_element(By.TAG_NAME, 'main').get_attribute('innerHTML')
            text = await get_processed_text(text, url)
            table_data = extract_data_from_page(text, str(CallSchema.model_json_schema()))
            table_data = re.sub(r"```json\s*|```", "", (table_data).strip())
            table_data = dict(json.loads(table_data))
            table_data_missing = {k: v for k, v in table_data.items() if v == ""}
            bases = table_data["bases"]
            download_pdf(bases, driver, options)
            pdf_data = {}
            for file in os.listdir(store_folder):
                if not file.endswith(".pdf"):
                    continue
                pdf_path = os.path.join(store_folder, file)
                pdf_jsondata, pdf_content = analyze_layout(pdf_path)
                pdf_data = query_pdf_data(json.dumps(pdf_jsondata), pdf_content, table_data)
                pdf_data = re.sub(r"```json\s*|```", "", (pdf_data).strip())+"\""
                pdf_data = safe_json_loads(pdf_data)
                table_data = {**table_data, **pdf_data[table_data_missing.keys()]}   
            call_data = table_data
            df.loc[i, call_data.keys()] = call_data.values()                 
        except (WebDriverException, NoSuchElementException) as e:
            print(f"Error al descargar el PDF: {e}")
        except PdfReadError as e:
            print(f"Error al leer el PDF: {e}")
        except Exception as e:
            print(f"Error al obtener los datos de las tablas: {e}")
        finally:
            call_data = table_data
            df.loc[i, call_data.keys()] = call_data.values() 
            driver.quit()
    
    return df


if __name__ == "__main__":
    asyncio.run(AgenticoSNPSAP())
    # print(json.dumps(AEISchema.model_json_schema()["properties"], indent=4))