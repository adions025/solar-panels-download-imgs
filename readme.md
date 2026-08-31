# Building Tiles Generator

Script para generar **tiles de edificios a partir de datos del Catastro**, procesando la información por municipio, provincia y rango de años.

Los datos catastrales se obtienen mediante la fuente `atom` y se procesan en paralelo utilizando varios workers.

## Uso

```bash
python thumbs_par.py \
  --city "<MUNICIPIO>" \
  --max-workers 4 \
  --year-ini 2012 \
  --year-end 2024 \
  --cadastre-source atom \
  --province "<PROVINCIA>"
```

## Parámetros

* `--city`: Municipio que se quiere procesar.
* `--province`: Provincia a la que pertenece el municipio.
* `--max-workers`: Número máximo de workers utilizados para procesar los datos en paralelo.
* `--year-ini`: Año inicial de los datos catastrales a procesar.
* `--year-end`: Año final de los datos catastrales a procesar.
* `--cadastre-source`: Fuente utilizada para obtener los datos del Catastro. Actualmente se utiliza `atom`.

## Ejemplos

### Barcelona

```bash
python thumbs_par.py --city "Sitges" --max-workers 4 --year-ini 2012 --year-end 2024 --cadastre-source atom --province "Barcelona"
```

### Lleida

```bash
python thumbs_par.py --city "Lleida" --max-workers 4 --year-ini 2012 --year-end 2024 --cadastre-source atom --province "Lleida"
```

### Tarragona

```bash
python thumbs_par.py --city "Reus" --max-workers 4 --year-ini 2012 --year-end 2024 --cadastre-source atom --province "Tarragona"
```

## Procesar un único año

Para generar los tiles correspondientes a un único año, se debe utilizar el mismo valor en `--year-ini` y `--year-end`.

```bash
python thumbs_par.py --city "Valldoreix" --max-workers 4 --year-ini 2018 --year-end 2018 --cadastre-source atom --province "Barcelona"
```

## Municipios procesados

Entre los municipios utilizados con este script se encuentran:

* Sitges
* Valldoreix
* Lleida
* Reus
* Cambrils
* Salou
* Calafell
* Tarragona

El municipio y la provincia pueden modificarse mediante los parámetros `--city` y `--province` según la zona que se quiera procesar.
