from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy, DFSDeepCrawlStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonXPathExtractionStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy, WebScrapingStrategy
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter
)

from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
import time


from aiofiles import open as aio_open
import aiohttp
import asyncio
import os
import json
import logging
import logging.config
import re

import pandas as pd
import numpy as np
from docling.document_converter import DocumentConverter

# cargamos configuacion de logging
logging.config.fileConfig('crawler/logconfig.conf')

# creamos logger
logger = logging.getLogger(__name__)

# cargamos variables de entorno
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv(".env")

async def write_log(log_file="crawler/logger.txt", result=None, regex: str = None):
    async with aio_open(log_file, "a", encoding="utf-8") as f:
        for res in result:
            if re.match(regex, res.url):
                    await f.write(res.extracted_content)
            else:
                    await f.write(res.url)
    

async def cienciaGob():
    
    LOG_FILE = "crawler/logger.txt"
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    URLS = URLS["cienciaGob"][:-1]
    
    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=[
            r"^https://www\.aei\.gob\.es/convocatorias/buscador-convocatorias",
        ]),

        # Only crawl specific domains
        # DomainFilter(
            # allowed_domains=["https://www.pap.hacienda.gob.es"],
        # ),

        # Only include specific content types
        ContentTypeFilter(allowed_types=["html"])
    ])
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        include_external=True,
        logger=logger,
        filter_chain=filter_chain
    )
    # TODO: extraer presupuesto
    schema = {
        "name": "Convocatoria",
        "baseSelector": "//div[@role='main']",    # Repeated elements
        "fields": [
            {
                "name": "convocatoria",
                "selector": ".//table[@class='table-striped table table-bordered ']/tbody/tr[1]/td/div",
                'type': "text",
                "default": ""
            },
            {
                "name": "plazos",
                "selector": ".//table[@class='table-striped table table-bordered ']/table/tbody/tr[6]/td",
                "type": "text",
                "default": ""
            },
            {
                "name": "entidad",
                "selector": ".//table[@class='table-striped table table-bordered ']/tbody/tr[11]/td/",
                "type": "text",
                "default": ""
            },
            {
                "name": "descripcion",
                "selector": ".//table[@class='table-striped table table-bordered ']/tbody/tr[7]/td/div/p[1]",
                "type": "text",
                "default": ""
            },
            {
                "name": "tipo",
                "selector": ".//table[@class='table-striped table table-bordered ']/tbody/tr[13]/td/",
                "type": "text",
                "default": ""
            },
            {
                "name": "presupuesto",
                "selector": ".//table[@class='table-striped table table-bordered ']/tbody/tr[9]/td/",
                "type": "text",
                "default": ""
            },
            {
                "name": "beneficiario",
                "selector": ".//table[@class='table-striped table table-bordered ']/tbody/tr[10]/td/",
                "type": "text",
                "default": ""
            },
            {
                "name": "bases",
                "selector": "//aside[@role='complementary']/div/div/div/div/ul/li[1]/a",
                "type": "attribute",
                "attribute": "href",
                "default": ""
            }
        ]
    }
    extraction_strategy = JsonXPathExtractionStrategy(schema=schema, logger=logger, verbose=True)
    browser_config = BrowserConfig(
        verbose=True,
        accept_downloads=True,
        java_script_enabled=True,
        ignore_https_errors=True,
    )
    contenido = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        #if not isinstance(convocatorias, list): # a instancia es una url
        i = 1
        while True:
            try:
                run_config = CrawlerRunConfig(
                    deep_crawl_strategy=deep_crawl_strategy,
                    scraping_strategy=LXMLWebScrapingStrategy(logger=logger),
                    extraction_strategy=extraction_strategy,
                    cache_mode=CacheMode.BYPASS,
                )
                result = await crawler.arun(url=URLS+str(i), config=run_config)
                if i > 1: #not len(result) > 1:
                    logger.info("Busqueda finalizada")
                    break
                logger.info("Cargando siguiente pagina")
                #await write_log(LOG_FILE, result, regex=r"^https://www.ciencia.gob.es/Convocatorias/*")
                contenido = [{**json.loads(res.extracted_content)[0], "url": res.url} for res in result if res.extracted_content != "[]"]
                i+=1  
            except Exception as e:
                logger.error(f"Error: {e}")

    return contenido


async def turismoGob():
    LOG_FILE = "crawler/logger.txt"
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    URLS = URLS["turismoGob"]    

    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=[
           r"^https://www\.mintur\.gob\.es/PortalAyudas/(?!.*Paginas)[\w\-]+",
           r"^https://www\.mintur\.gob\.es/PortalAyudas/[\w\-]+/DescripcionGeneral/Paginas/Index.aspx"
        ]),

        ## Only include specific content types
        ContentTypeFilter(allowed_types=["aspx"])
    ])
    deep_crawl_strategy = DFSDeepCrawlStrategy(
        max_depth=1,
        include_external=False,
        logger=logger,
        filter_chain=filter_chain
    )
    schema = {
        "name": "Convocatoria",
        "baseSelector": "//div[contains(@class, 'interior-container')]",
        "fields": [
            {
                "name": "convocatoria",
                "selector": ".//h1",
                'type': "text",
                "default": ""        
            },
            {
                "name": "fecha_publicación",
                "selector": ".//dl[@class='datos-ayuda']/dd[1]",
                'type': "text",
                "default": ""        
            },
            {
                "name": "plazos",
                "selector": ".//dl[@class='datos-ayuda']/dd[2]",
                "type": "text",
                "default": ""        
            },
            {
                "name": "entidad",
                "selector": ".//div[@class='lista-gestion']/h2",
                "type": "text",
                "default": ""        
            }
        ]
    }
    extraction_strategy = JsonXPathExtractionStrategy(schema=schema, logger=logger, verbose=True)
    browser_config = BrowserConfig(
        verbose=True,
        accept_downloads=True,
        java_script_enabled=True,
        ignore_https_errors=True,
    )
    load_more_js = [
        "window.scrollTo(0, document.body.scrollHeight);",
        # The "More" link at page bottom
        "document.querySelector('a[href=\"javascript:void(0)\"]')?.click();"  
    ]    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        #if not isinstance(convocatorias, list): # a instancia es una url
        try:
            run_config = CrawlerRunConfig(
                deep_crawl_strategy=deep_crawl_strategy,
                scraping_strategy=LXMLWebScrapingStrategy(logger=logger),
                extraction_strategy=extraction_strategy,
                #cache_mode=CacheMode.ENABLED,
                #js_code=load_more_js,
                #wait_for="""js:(() => {
                #    return document.querySelectorAll('div.listado-consulta ul li').length > 10;
                #})""",
                # Mark that we do not re-navigate, but run JS in the same session:
                #js_only=True,
                #session_id="hn_session"
            )
            result1 = await crawler.arun(url=URLS, config=run_config)
            contenido = [{**json.loads(res.extracted_content)[0], "url": res.url} for res in result1[1:]]
            for entrada in contenido:
                r = requests.get(os.path.join(entrada["url"], "DescripcionGeneral/Paginas/Index.aspx"))
                if r.status_code != 200:
                    r = requests.get(os.path.join(entrada["url"], "DescripcionGeneral/Paginas/descripcion.aspx"))
                pool = BeautifulSoup(r.content, 'html.parser')
                descripcion = " ".join([p.get_text(strip=True) for p in pool.select(".col-contenido p")])
                entrada["descripcion"] = descripcion
            logger.info("Informacion cargada")
            #await write_log(LOG_FILE, result1, regex=r"^https://www.mintur.gob.es/PortalAyudas/[\w]+/Paginas/Index.aspx")
        except Exception as e:
            logger.error(f"Error: {e}")

    return contenido


async def SNPSAP():
    pdf_filename = "listado4_11_2025.pdf" #f"listado{datetime.today().month}_{datetime.today().day}_{datetime.today().year}.pdf" 
    LOG_FILE = "crawler/logger.txt"
    
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    base_url = URLS["SNPSAP"]

    if pdf_filename not in os.listdir("./downloads/pdf"):

        # Configure Firefox options for automatic PDF downloading
        firefox_options = Options()
        firefox_options.set_preference("browser.download.folderList", 2)  # Use custom directory
        firefox_options.set_preference("browser.download.dir", os.path.join(os.getcwd(), "downloads/pdf"))  # Change this path
        firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")  # Auto-save PDFs
        firefox_options.set_preference("pdfjs.disabled", True)  # Disable PDF preview in browser
        firefox_options.set_preference("headless", True)
    
        driver = webdriver.Firefox(options=firefox_options)

        # Open the target webpage
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

        except Exception as e:
            print("Error: PDF link not found.", str(e))
    
    input_doc_path = Path(os.path.join('downloads/pdf', pdf_filename))
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)
    df = pd.DataFrame()
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        df = pd.concat([df, table_df], ignore_index=True, axis=0)
        logging.info(f"Data extraída de tabla {table_ix}")
    df["url"] = list(map(lambda x: os.path.join(base_url, x), df["Código BDNS"]))
    df[["presupuesto", "fecha_inicio", "fecha_final", "finalidad"]] = pd.DataFrame(
        data=[],
        columns=["presupuesto", "fecha_inicio", "fecha_final", "finalidad"],
        dtype=str
    )
    schema = {
        "name": "Convocatoria",
        "baseSelector": "//app-convocatoria",    # Repeated elements
        "fields": [
            {
                "name": "presupuesto",
                "selector": "//*[@id='print']/div[2]/div[5]/div[2]",
                'type': "text",
                "default": ""        
            },
            {
                "name": "fecha_inicio",
                "selector": "//*[@id='cdk-accordion-child-1']/div/div[1]/div[2]/div[3]/div[2]",
                'type': "text",
                "default": ""        
            },
            {
                "name": "fecha_final",
                "selector": "//*[@id='cdk-accordion-child-1']/div/div[1]/div[2]/div[4]/div[2]",
                "type": "text",
                "default": ""        
            },
            {
                "name": "finalidad",
                "selector": "//*[@id='print']/div[4]/div[4]/div[2]",
                'type': "text",
                "default": ""        
            },
            {
                "name": "localidad",
                "selector": "//*[@id='print']/div[4]/div[3]/div[2]/div",
                "type": "text",
                "default": ""        
            },
            {
                "name": "tipo",
                "selector": "//*[@id='print']/div[2]",
                "type": "text",
                "default": ""        
            },
            {
                "name": "bases",
                "selector": "//*[@id='print']/div[5]/div[2]/div[2]/a",
                "type": "attribute",
                "attribute": "href",
                "default": ""                
            }
        ]
    }
    extraction_strategy = JsonXPathExtractionStrategy(schema=schema, logger=logger, verbose=True)
    run_config = CrawlerRunConfig(
        scraping_strategy=WebScrapingStrategy(logger=logger),
        extraction_strategy=extraction_strategy,
    )
    browser_config = BrowserConfig(
         verbose=True,
         accept_downloads=True,
         ignore_https_errors=True,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        #if not isinstance(convocatorias, list): # a instancia es una url
        for i, url in enumerate(df["url"]):
            try:
                result = await crawler.arun(url=url, config=run_config) 
                if result.success:
                    data = dict(json.loads(result.extracted_content)[0])
                    df.loc[i, data.keys()] = data.values()
                    #async with aio_open(LOG_FILE, "a", encoding="utf-8") as f:
                    #    await f.write(json.dumps(data, indent=4))
                #await write_log(LOG_FILE, result1, regex=r"^https://www.mintur.gob.es/PortalAyudas/[\w]+/Paginas/Index.aspx")
            except Exception as e:
                logger.error(f"Error: {e}")
    
    return df


if __name__ == "__main__":
    df = asyncio.run(SNPSAP())
    print(df.head())
    print(df.info())