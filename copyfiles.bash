#!/bin/bash
# usage: copyfiles.bash <destination_folder> <new_folder_name>

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <destination_folder> <new_folder_name>"
  exit 1
fi

# Assign arguments to variables
DESTINATION_FOLDER=$1
NEW_FOLDER_NAME=$2

# Check if the data folder exists
if [ ! -d "./data" ]; then
  echo "The 'data' folder does not exist in the current directory."
  exit 1
fi

# Create the new folder inside the destination folder
mkdir -p "$DESTINATION_FOLDER/$NEW_FOLDER_NAME"

# Copy the data folder to the new folder
cp -r ./data "$DESTINATION_FOLDER/$NEW_FOLDER_NAME"
cp ./generatehdf5.bash ./call_this_bash.py ./map_recreate.py "$DESTINATION_FOLDER/$NEW_FOLDER_NAME"

echo "Files have been copied to '$DESTINATION_FOLDER/$NEW_FOLDER_NAME/data'."
