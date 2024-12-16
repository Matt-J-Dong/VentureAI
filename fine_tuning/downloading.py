import gdown

# URL of the file
url = "https://docs.google.com/uc?export=download&id=1lnfI5u0EOLpdXLNnIF_-0giAxctJ6QXh"

# Output file name
output = "downloaded_file.zip"  # Change this to your preferred file name

# Download the file
gdown.download(url, output, quiet=False)

print(f"File downloaded as {output}")