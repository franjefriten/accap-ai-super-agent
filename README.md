# Agente ACCAP
Este es un proyecto de un Super Agente IA que capta información de convocatorias de ayudas y las sirve de forma amigable al usuario. Emplea crawl4ai para obtener datos mediante webscrapping + IA. Usa servicios de Azure AI para su ejecución. Se sirve al usuario a través de una aplicación Streamlit

## INSTALACIÓN
Este software se instala y lanza a partir de Docker. 
En primer lugar
```
git clone https://github.com/franjefriten/tfm-ai-super-agent
```
Ejecútese desde el directorio raíz del proyecto
```
docker compose up -d --build
```
Y accedase a `localhost:8080`

### OBTENCIÓN DE DATOS

A partir de aquí, se dan opciones
Métodos:
  * clasico: Webscrapping clásico
  * agentico: Webscrapping mediante un modelo de IA

Fuentes:
  * SNPSAP: Sistema Nacional de Publicidad de Subvenciones y Ayudas Públicas
  * cienciaGob: Ministerio de Ciencia e Investigación (solo admite clásico)
  * turismoGob: Ministerio de Turismo e Industria (solo admite clásico)
  * AEI: Agencia Estatal de Investigación

Para rellenar con registros la base de datos PostgreSQL, existen dos opciones.

**OPCIÓN 1**
El sistema tiene habilitados dos botones para obtener datos desde 
AEI agentico y SNPSAP agentico, que son las principales. Es importante recordar
que es *imprescindible* pulsar uno de estos botones primero para captar información
con la que consultar. Puede tardar hasta un minuto o más para obtener todo el proceso

**OPCIÓN 2**
Ejecútese en una terminal los siguientes comandos desde una terminal.
```
~$ virtualenv venv
~$ source ./venv/bin/activate
(venv)~$ pip install -r requirements2.txt
(venv)~$ pip install --no-deps crawl4ai==0.5.0.post8
(venv)~$ python3 get_and_store_data_from_source.py
```
y seleccione las opciones que aparece en pantalla por la terminal.

## AUTORÍA
Francisco Jesús Frías Tenza
