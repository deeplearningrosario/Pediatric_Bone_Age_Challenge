#!/bin/bash -
#===============================================================================
#
#          FILE: downloader_dataset.sh
#
#         USAGE: ./downloader_dataset.sh
#
#   DESCRIPTION: Dataset download
#
#  REQUIREMENTS: curl, unzip
#        AUTHOR: MARANI MATIAS EZEQUIEL,
#  ORGANIZATION:
#       CREATED: 03/08/18 08:39
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

# Dataset 209 img, 106 male and 103 famele
dataset="https://doc-0k-4c-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/p6j0socede0t3ep0q92gfprjogi786v9/1533290400000/03448145893575284703/*/1DeMVpdFLya8IOAyP31tv6WXr5Kn-rVcG?e=download -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: en-GB,en;q=0.5' --compressed -H 'Referer: https://drive.google.com/' -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'Pragma: no-cache' -H 'Cache-Control: no-cache'"

curl $(echo "$dataset") -o dataset.zip && unzip dataset.zip
