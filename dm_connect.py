import pymysql
import csv
# Database credentials from env.php
host = '68.66.194.91'
dbname = 'drew_mage10dec'
username = 'drew_mage433'
password = '34Vp))90Sh'

# Create a cursor object to execute SQL queries
try:
    # Establish a connection to the database
    conn = pymysql.connect(
        host=host,
        user=username,
        password=password,
        database=dbname
    )
    cursor = conn.cursor()
except:
    print("Connection failed")
    cursor = None

def find_entities(cursor=cursor):
    # List of attribute codes
    attribute_codes = ["length", "voltage", "price"]

    # Base SQL query
    sql = """
    SELECT e.entity_id as product_id, e.sku
    """

    # Add attribute codes to the SELECT clause
    for code in attribute_codes:
        sql += f", v_{code}.value as {code}"

    # Add FROM clause
    sql += """
    FROM
        mg6t_catalog_product_entity as e
    """

    # Add attribute codes to the JOIN clause
    for code in attribute_codes:
        sql += f"""
        LEFT JOIN
            mg6t_catalog_product_entity_varchar as v_{code}
            ON e.entity_id = v_{code}.entity_id
            AND v_{code}.attribute_id = (SELECT attribute_id FROM mg6t_eav_attribute WHERE attribute_code = '{code}' AND entity_type_id = 4)
        """
    # Example usage:
    # get_all_products(cursor)


    # Execute the SQL query
    cursor.execute(sql)

    # Fetch the results
    results = cursor.fetchall()

    # Process the results
    for row in results:
        print(row)


    # Close the cursor and connection
    cursor.close()
    conn.close()

def get_all_products(cursor):
    # Get attribute_id for 'name'
    sql = """
    SELECT attribute_id 
    FROM mg6t_eav_attribute 
    WHERE attribute_code = 'name' 
    AND entity_type_id = (SELECT entity_type_id 
                          FROM mg6t_eav_entity_type 
                          WHERE entity_type_code = 'catalog_product')
    """
    cursor.execute(sql)
    result = cursor.fetchone()
    attribute_id = result[0]  # Adjusted to work with tuple

    # Get product names
    sql = f"SELECT value FROM mg6t_catalog_product_entity_varchar WHERE attribute_id = {attribute_id}"
    cursor.execute(sql)
    product_names = cursor.fetchall()

    with open('product_names.csv', 'w', newline='', encoding='utf-8') as f:  # Added encoding='utf-8'
        writer = csv.writer(f)
        writer.writerow(['Product Name'])
        for row in product_names:
            writer.writerow([row[0]])  # Adjusted to work with tuple


def get_product_by_name_or_sku(search_term, cursor=cursor):
    # Create a new cursor that returns results as a dictionary
    dict_cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Get attribute_id for 'name'
    sql = """
    SELECT attribute_id 
    FROM mg6t_eav_attribute 
    WHERE attribute_code = 'name' 
    AND entity_type_id = (SELECT entity_type_id 
                          FROM mg6t_eav_entity_type 
                          WHERE entity_type_code = 'catalog_product')
    """
    dict_cursor.execute(sql)
    result = dict_cursor.fetchone()
    attribute_id = result['attribute_id']  # Adjusted to work with dictionary

    # Search for products by name or SKU
    sql = f"""
    SELECT e.*, u.request_path as url 
    FROM mg6t_catalog_product_entity as e
    LEFT JOIN mg6t_url_rewrite as u ON e.entity_id = u.entity_id
    WHERE e.entity_id IN (
        SELECT entity_id 
        FROM mg6t_catalog_product_entity_varchar 
        WHERE attribute_id = {attribute_id} 
        AND value LIKE '%{search_term}%'
    ) OR sku LIKE '%{search_term}%'
    ORDER BY sku 
    LIMIT 1
    """
    dict_cursor.execute(sql)
    product = dict_cursor.fetchone()

    # Close the dictionary cursor
    dict_cursor.close()

    return product

def execute_query(sql, cursor=cursor):
    print("Executing query: " + sql)
    cursor.execute(sql)
    results = cursor.fetchall()
    return results

# EXTRACT WARRANTY INFO
# BY MFG
# IF NO MFG:
    # WHAT MFG? 
    # CHECK_WARRANTY (MFG_NAME)
# TECHNICAL PRODUCT SPECS
    # IF NO PRODUCT
    # WHAT SKU NUMBER?
    # CHECK PRODUCT SPECS (SKU, PRODUCT_NAME) 


# get_all_products(cursor)

# print("https://directmachines.com/"+str(get_product_by_name_or_sku("doall")['url']))