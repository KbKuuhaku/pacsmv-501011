data_root="data/uno-cards"
file_name="8T9Ltfa28U?key=5jQ1oDVhFB"

cd ${data_root}

if [ -f "data.yaml" ]; then
    echo "files have been downloaded to ${data_root}, finished"
    cd -
    exit 1
fi

# Roboflow already provides the YOLOv11 format for developers to download
download_url="https://universe.roboflow.com/ds/${file_name}"

wget ${download_url}

# zip files 
unzip ${file_name}

rm ${file_name}

echo "Finished"

cd -
