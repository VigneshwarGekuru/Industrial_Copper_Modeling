import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
from streamlit_option_menu import option_menu
import re

import base64
from PIL import Image

img = Image.open("sucesshand.png")
st.set_page_config(page_icon = img,layout = "wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#ffa500 ;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)


def get_img_as_base64(file):
    with open(file,"rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_1= get_img_as_base64("wp12022834.jpg")

hide_st_style = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;} 
header {visibility:hidden;} 
</style> 
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

page_bg_img = f'''
<style> 
[data-testid="stAppViewContainer"] {{
background-image :url("data:image/png;base64,{img_1}");
background-size : cover;
}}

[data-testid="stHeader"] {{
background:rgba(0,0,0,0);  
}}

[data-testid="stSidebar"] {{
    
background-image :url("data:image/png;base64,{img_1}");
background-size : cover;
    
}}

</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


with st.sidebar:
    SELECT = option_menu(
        menu_title = None,
        options = ["ABOUT","PREDICT SELLING PRICE","PREDICT STATUS","CONTACT"],
        default_index=1,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "white","size":"cover"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "#ffa500"},
            "nav-link-selected": {"background-color": "#ffa500"},}
        )

# Define the possible values for the dropdown menus

status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product=['611728', '628112', '628117', '628377', '640400', '640405', '640665', 
              '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
              '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
              '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
              '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']




if SELECT == "ABOUT": 
  st.write('##  In this machine learning project, the main objective is to predict the selling price and status of an item or product based on the provided values of specific features. ')
  st.write('### Predicting Selling Price: The project aims to build a model that can estimate the selling price of an item.This could be useful for various scenarios such as e-commerce platforms, real estate, or used car marketplaces. The model will take into account various features or attributes of the item, such as quantity, thickness, width, location, or any other relevant factors that influence the selling price. ')
  st.write('### Predicting Status: Along with predicting the selling price, the project also focuses on predicting the status of the item. This refers to whether the item is likely to be sold or not. The model will consider the same set of features and use them to determine the likelihood of the item being sold based on historical data or patterns observed in the dataset.If it is sold status will be considered as won else status will be considered as lost')


if SELECT == "PREDICT SELLING PRICE":     

        # Define the widgets for user input
        
        st.write('## PREDICT SELLING PRICE')
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)

            with col3:               
                st.write( f'<h5 style="color:rgb(255, 165, 0,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:1 & Max:1000000000)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.balloons()
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #ffa500;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
             
        if submit_button and flag==0:
            
            import pickle
            

            dn = pd.DataFrame({'customer':[float(customer)], 'country':[country], 'status':[status], 'item type':[item_type], 'application':[int(application)], 'width':[float(width)],
                               'product_ref':[int(product_ref)], 'quantity tons_log':[np.log(float(quantity_tons))], 'thickness_log':[np.log(float(thickness))]})

            dn_one_hot = pd.get_dummies(dn, columns=['application'], prefix='application')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['country'], prefix='country')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['status'], prefix='status')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['product_ref'], prefix='product_ref')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['item type'], prefix='item type')


            with open('X_f.columns.pickle', 'rb') as a:
                k = pickle.load(a)
            new_data = pd.DataFrame(columns=k)       # Getting all columns of original dataframe
        
            new_data = new_data.append(dn_one_hot)     # appending dn_one_hot to new_data
            new_data = new_data.fillna(0)                         # filling nan values with '0'
            features = new_data.values

            with open(r'Scalar.pickle', 'rb') as b:
                l = pickle.load(b)
            new_sample = l.transform(features)


            with open("DTR_model.pickle", 'rb') as c:
                m = pickle.load(c) 
            new_pred = m.predict(new_sample)
            
            st.write('## :green[Predicted selling price:] ')

            def write_with_larger_font(value):
                # Increase the font size using HTML tags
                
                larger_font = "<h1 style='font-size: 30px;'>{}</h1>".format(value)
                st.write(larger_font, unsafe_allow_html=True)

            write_with_larger_font(int(np.exp(new_pred)))
            
                        
            
if SELECT == "PREDICT STATUS":     
    
        st.write('## PREDICT STATUS')
        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:1 & Max:1000000000)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
                st.balloons()
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #ffa500;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)


    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
             
        if csubmit_button and cflag==0:
            import pickle
            with open(r"DTC_model.pickle", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'C_Scalar.pickle', 'rb') as f:
                cscaler_loaded = pickle.load(f)

                        
            dn = pd.DataFrame({'customer':[int(ccustomer)], 'country':[ccountry],'selling_price_log':[np.log(float(cselling))], 'item type':[citem_type], 'application':[int(capplication)], 'width':[float(cwidth)],
                               'product_ref':[int(cproduct_ref)], 'quantity tons_log':[np.log(float(cquantity_tons))], 'thickness_log':[np.log(float(cthickness))]})
            
            dn_one_hot = pd.get_dummies(dn, columns=['application'], prefix='application')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['country'], prefix='country')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['product_ref'], prefix='product_ref')
            dn_one_hot = pd.get_dummies(dn_one_hot, columns=['item type'], prefix='item type')
            
            with open('X_u.columns.pickle', 'rb') as a:
                 k = pickle.load(a)
            new_data = pd.DataFrame(columns=k)       # Getting all columns of original dataframe  

            new_data = new_data.append(dn_one_hot)              # appending dn_one_hot to new_data         
            new_data = new_data.fillna(0)                         # filling nan values with '0'
            features = new_data.values

            with open('C_Scalar.pickle', 'rb') as b:
                l = pickle.load(b)   
            new_sample = l.transform(features)
            
            
            with open('DTC_model.pickle', 'rb') as c:
                 m = pickle.load(c)
            new_pred = m.predict(new_sample)  
            if new_pred==1:
                st.write('## :green[The Status is Won]')
            else:
                st.write('## :red[The status is Lost]')


if SELECT == "CONTACT": 

    st.title("About me")
    name = "Vigneshwar Gekuru"
    mail = (f'{"Mail Me At :"}  {"vigneshwargekuru@gmail.com"}')
    description = "An aspiring DATA-SCIENTIST with brilliant ideas"

    col1,col2,col3= st.columns(3)
    col2.image(Image.open("my.png"),width = 240)
    with col1:
        st.header(name)
        st.write(description)
        st.write(mail)
        st.image("mail.png",width = 50)
    with col3:
      st.image("sucesshand.png",width=250)
      st.write("###### If you like my work kindly reach out to me by given links")
        

    col1,col2,col3,col4,col5 = st.columns(5)

    col1.write('[Twitter](https://twitter.com/G_Vigneshwar_01)')
    col1.image(Image.open("twitter.png"),width = 200)
    col2.write('[GIT-Hub](https://github.com/VigneshwarGekuru)')
    col2.image(Image.open("Github.png"),width = 170)
    col3.write('[linkedIN](https://www.linkedin.com/in/vigneshwar-gekuru-6834a5240/)')
    col3.image(Image.open("Linkedin.png"),width = 200)
    col4.write('[Facebook]()')
    col4.image(Image.open("facebook.png"),width = 200)
    col5.write('[Instagram]()')
    col5.image(Image.open("instagram.png"),width = 200)



st.write('<h6 style="color:rgb(255, 165, 0, 0.35); font-size: 25px;">App Created by Vigneshwar Gekuru</h6>', unsafe_allow_html=True)


