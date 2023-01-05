

#### Steps

node fetch-urls.mjs -> scrapes and finds all the cdls

python download_dlcs.py -> uses the urls to download all the dlcs

python check_magic.py -> checks magic and then unzips if required

python move_unknown_files.py -> moves all the files that are not cdlc files to Misc folder

PSARCExtractor.exe -> from https://github.com/draguve/PSARCExtractor extracts psarc files to folders and then deletes the psarc files

python convert_wems.py -> convert wem files in those folders to ogg files