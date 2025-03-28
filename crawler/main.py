from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonXPathExtractionStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter
)
from aiofiles import open as aio_open
import aiohttp
import xml.etree.ElementTree as ET
import asyncio
import os
import json
import logging
import logging.config
from asyncio import Semaphore
import re

# cargamos configuacion de logging
logging.config.fileConfig('crawler/logconfig.conf')

# creamos logger
logger = logging.getLogger(__name__)

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
            r"^https://www\.ciencia\.gob\.es/Convocatorias/",
        ]),

        # Only crawl specific domains
        # DomainFilter(
            # allowed_domains=["https://www.pap.hacienda.gob.es"],
        # ),

        # Only include specific content types
        ContentTypeFilter(allowed_types=["html"])
    ])
    deep_crawl_strategy = BestFirstCrawlingStrategy(
        max_depth=1,
        include_external=True,
        logger=logger,
        filter_chain=filter_chain
    )
    schema = {
        "name": "Convocatoria",
        "baseSelector": "//div[@id='convocatoriasEmpleoActivas']",    # Repeated elements
        "fields": [
            {
                "name": "convocatoria",
                "selector": ".//h1",
                'type': "text"
            },
            {
                "name": "fecha_publicación",
                "selector": ".//p[@class='fecha']/span",
                'type': "text"
            },
            {
                "name": "plazos",
                "selector": ".//p[@class='plazos']",
                "type": "text"
            },
            {
                "name": "entidad",
                "selector": "//p[@class='organoInstructor']/span",
                'type': "text"
            },

        ]
    }
    extraction_strategy = JsonXPathExtractionStrategy(schema=schema, logger=logger, verbose=True)
    browser_config = BrowserConfig(
        verbose=True,
        accept_downloads=True,
        java_script_enabled=True,
        ignore_https_errors=True,
    )
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
                if not len(result) > 1:
                    logger.info("Busqueda finalizada")
                    break
                logger.info("Cargando siguiente pagina")
                #await write_log(LOG_FILE, result, regex=r"^https://www.ciencia.gob.es/Convocatorias/*")
                i+=1  
            except Exception as e:
                logger.error(f"Error: {e}")


async def turismoGob():
    LOG_FILE = "crawler/logger.txt"
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    URLS = URLS["turismoGob"]    

    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=[
           r"^https://www\.mintur\.gob\.es/PortalAyudas/(?!.*Paginas)[\w\-]+",
        ]),

        ## Only include specific content types
        ContentTypeFilter(allowed_types=["aspx"])
    ])
    deep_crawl_strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        include_external=False,
        logger=logger,
        filter_chain=filter_chain
    )
    schema = {
        "name": "Convocatoria",
        "baseSelector": "//div[@class='interior-container home-ayuda']",    # Repeated elements
        "fields": [
            {
                "name": "convocatoria",
                "selector": ".//h1",
                'type': "text"
            },
            {
                "name": "fecha_publicación",
                "selector": ".//dl[@class='datos-ayuda']/dd[0]",
                'type': "text"
            },
            {
                "name": "plazos",
                "selector": ".//dl[@class='datos-ayuda']/dd[1]",
                "type": "text"
            },
            {
                "name": "entidad",
                "selector": "//div[@class='lista-gestion']/h2",
                'type': "text"
            },

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
            print(result1[0].extracted_content)
            logger.info("Informacion cargada")
            await write_log(LOG_FILE, result1, regex=r"^https://www.mintur.gob.es/PortalAyudas/[\w]+/Paginas/Index.aspx")
        except Exception as e:
            logger.error(f"Error: {e}")


async def test():
    LOG_FILE = "crawler/logger.txt"
    with open("crawler/urls.json", "rb") as urlsfile:
        URLS = dict(json.load(urlsfile))
    URLS = URLS["turismoGob"]    

    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=[
           r"^https://www\.mintur\.gob\.es/PortalAyudas/(?!.*Paginas)[\w\-]+",
        ]),

        ## Only include specific content types
        ContentTypeFilter(allowed_types=["aspx"])
    ])
    deep_crawl_strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        include_external=False,
        logger=logger,
        filter_chain=filter_chain
    )
    schema = {
        "name": "Convocatoria",
        "baseSelector": "//div[@class='interior-container home-ayuda']",    # Repeated elements
        "fields": [
            {
                "name": "convocatoria",
                "selector": ".//h1",
                'type': "text"
            },
            {
                "name": "fecha_publicación",
                "selector": ".//dl[@class='datos-ayuda']/dd[0]",
                'type': "text"
            },
            {
                "name": "plazos",
                "selector": ".//dl[@class='datos-ayuda']/dd[1]",
                "type": "text"
            },
            {
                "name": "entidad",
                "selector": "//div[@class='lista-gestion']/h2",
                'type': "text"
            },

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
            print(result1[0].extracted_content)
            logger.info("Informacion cargada")
            await write_log(LOG_FILE, result1, regex=r"^https://www.mintur.gob.es/PortalAyudas/[\w]+/Paginas/Index.aspx")
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())