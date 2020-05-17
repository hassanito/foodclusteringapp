import os

import urllib.request

Email = "hassanitohajj@gmail.com"
Password = "gimzjzkyjbgneuzo"
Recipient = "hassanlhage@hotmail.com"
class Product:
    def __init__(self,ID,images):
        self.product_ID = ID
        self.product_images_URL = images
        self.number_of_images = len(images)
filepath = 'C:\\Users\\hassanelhajj\\Desktop\\docs2\\fyp\\productsImagesURLs.txt'
new_filepath = 'C:\\Users\\hassanelhajj\\Desktop\\docs2\\fyp\\newfile.txt'

products ={}
#products is a hashmap with key=ID and value = list of url images of product Id
with open(new_filepath) as fp:
   line = fp.readline()
   cnt = 0
   lines = []
   while line:
       #print("Line {}: {}".format(cnt, line.strip()))
       line = fp.readline()
       lines.append(line)
       id = lines[cnt][:14]
       url = lines[cnt][15:]
       if(id in products):
           products[id].append(url)
       else:
           products[id] = [url]
       cnt += 1

def send_email(user, pwd, recipient, subject, body):
    import smtplib

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user,pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print ('successfully sent the mail')
    except:
        print ("failed to send mail")

def CombinedList(list1,list2):
    set_1 = set(list1)
    set_2 = set(list2)
    list_2_items_not_in_list_1 = list(set_2 - set_1)
    combinedList = list1 + list_2_items_not_in_list_1
    return combinedList

def GetProductsFromFile(filename):
    products ={}
    #products is a hashmap with key=ID and value = list of url images of product Id
    with open(filename) as fp:
        line = fp.readline()
        cnt = 0
        lines = []
        while line:
            #print("Line {}: {}".format(cnt, line.strip()))
            lines.append(line)
            line = fp.readline()
            id = lines[cnt][:14]
            url = lines[cnt][15:]
            if(id in products):
                products[id].append(url)
            else:
                products[id] = [url]
            cnt += 1
    return products


def MergeDictionnaries(dicFile1,dicFile2):
    products = {}
    for i in dicFile1:
        for j in dicFile2:
            if(i not in dicFile2):
                if(i not in products):
                    products[i]=dicFile1[i]
            elif(j not in dicFile1):
                if(j not in products):
                    products[j]=dicFile2[j]
            else:
                if(i not in products and j not in products):
                    products[i]=CombinedList(dicFile1[i],dicFile2[j])
    print(products.keys())
    return products

def ProductDictionnaryToFile(filepath,productsDictionary):
    fp = open(filepath,mode='w')
    for i in productsDictionary:
        for j in productsDictionary[i]:
            fp.write(str(i)+' '+str(j)+'\n')
    fp.close()

def ProductDictionnaryToFileNoSpace(filepath,productsDictionary):
    fp = open(filepath,mode='w')
    for i in productsDictionary:
        for j in productsDictionary[i]:
            fp.write(str(i)+' '+str(j))
    fp.close()

def MergeFiles(oldFile,newFile):
    oldProducts = GetProductsFromFile(oldFile)
    newProducts = GetProductsFromFile(newFile)
    products = MergeDictionnaries(oldProducts,newProducts)
    ProductDictionnaryToFile(oldFile,products)

#this function returns the list of files in this folder
def GetListFileContent(FileListLocation):
    os.chdir(FileListLocation)
    return os.listdir(".")

def GetFileName(FileListLocation):
    FolderContent = GetListFileContent(FileListLocation)
    if(len(FolderContent)==0):
        return "ProductList_1.txt"
    else:
        return "ProductList_2.txt"

    #after this function do another task to train the model

#this function checks if there's 2 list files in the folder and merges them if needed
def MergeCheck(FolderLocation):
    folderContent =GetListFileContent(FolderLocation)
    folderSize = len(folderContent)
    if(folderSize==0):
        pass
    elif(folderSize==1):
        pass
    else:
        File1 = FolderLocation +"\\ProductList_1.txt"
        File2 = FolderLocation +"\\ProductList_2.txt"
        MergeFiles(File1,File2)
        os.remove(File2)
        send_email(Email,Password,Recipient,"Merging Products Lists", "Dear user, your products lists were merged")
        #SEND EMAIL THAT FILES WERE MERGED

def Train(number):
    import time
    send_email(Email,Password,Recipient,"Training model ", "Dear user your model has started trianing")
    time.sleep(number)
    print("Training done")
    send_email(Email,Password,Recipient,"Training model ", "Dear user your model has finished trianing")
products_objects = {}
#products_objects is a hashmap with key =ID and value = object Product with ID
for product in products.keys():
    p = Product(product,products[product])
    products_objects[product]=p

#clean_data(products_objects,"auchan")

#products_objects_number_images is a hashmap with key = ID
products_objects_number_images = {}
for id in products_objects:
    products_objects_number_images[id]=products_objects[id].number_of_images
#counts the objects with the number of images
stats = {}
for id in products_objects_number_images:
    number = products_objects_number_images[id]
    if(number in stats):
        stats[number] = stats[number]+1
    else:
        stats[number]=1
def find_ids_with_images(number_of_images):
    ids=[]
    for id in products_objects_number_images:
        number=products_objects_number_images[id]
        if(number_of_images==number):
            ids.append(id)
    return ids


def image_installer(url,picture_filename):
    try:
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', 'whatever')
        filename, headers = opener.retrieve(url, picture_filename)
        return 1
    except:
        print("no picture")
        return 0


def image_installer_by_id(id):

    try:

        import os
        os.chdir("C:\\users\\hassanelhajj\\desktop\\docs2\\fyp\\new_client_data_1")
        cur_dir= os.getcwd()+"\\"+str(id)
        if(not os.path.exists(cur_dir)):
            os.mkdir(str(id))
        os.chdir(cur_dir)
        product = products_objects[id]
        urls = product.product_images_URL
        j=0
        for i in urls:
            j = j+1
            print(i)
            image_installer(i.split('\n')[0],str(j)+'.jpg')
        print("done")
        return 1
    except:
        print("already exists")
        return 0

def install_images(min_number_of_images):
    id_list = find_ids_with_images(min_number_of_images)
    for id in id_list:
        image_installer_by_id(id)
    print("DONE")

def remove_elements(prod,keyword):

    x= []

    prod = products_objects[prod]
    for i in prod.product_images_URL:

        if(keyword in i):
            x.append(i)
    for i in x:
        prod.product_images_URL.remove(i)
    return prod

def clean_data(prod_obj,keyword):

    for i in prod_obj:

        remove_elements(str(i),keyword)

#clean file was already used to clean the data so don't run this function again
def clean_file(file):
    bad_words = ['auchan','leclercdrive']

    with open(file) as oldfile, open('C:\\Users\\hassanelhajj\\Desktop\\fyp\\newfile.txt', 'w') as newfile:
        print(newfile)
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)


