import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(tablename):
    '''
    input desired table name as a string
    returns table as pandas data frame
    '''
    df = pd.read_csv(f'../data/ga_archive/{tablename}', sep = '|')
    return df

def combine_dfs(df1, df2):
    new_lst = list(df2['registration_number'])
    mask = df1['registration_number'].isin(new_lst)
    df1['new_registration'] = mask
    df1['new_registration'] = df1['new_registration'].map({True: int(1), False: int(0)})
    return df1

def clean_data(df1):
    df1 = df1.drop([
    'land_district',
    'land_lot',
    'status_reason',
    'city_precinct_id',
    'county_districta_name',
    'county_districta_value',
    'county_districtb_name',
    'county_districtb_value',
    'city_dista_name',
    'city_dista_value',
    'city_distb_name',
    'city_distb_value',
    'city_distc_name',
    'city_distc_value',
    'city_distd_name',
    'city_distd_value',
    'party_last_voted',
    'city_school_district_name',
    'municipal_name',
    'municipal_code',
    'ward_city_council_code',
    'race_desc',
    'residence_city',
    'residence_zipcode',
    'county_precinct_id',
    'city_school_district_value',
    'senate_district',
    'house_district',
    'judicial_district',
    'commission_district',
    'school_district',
    'date_added',
    'date_changed',
    'district_combo',
    'last_contact_date',
    'ward_city_council_name',
    'date_last_voted',
    'registration_date'
    ], axis=1)

    #df1 = df1.set_index(df1['registration_number'], drop=True)
    df1 = df1.drop(['registration_number'], axis=1)

    df1['voter_status'] = df1['voter_status'].map({'A': int(1), 'I': int(0)})

    r_dummies = pd.get_dummies(df1['race'], dtype='int64')
    df1[r_dummies.columns] = r_dummies
    df1 = df1.drop(['race'], axis=1)

    g_dummies = pd.get_dummies(df1['gender'], dtype='int64')
    df1[g_dummies.columns] = g_dummies
    df1 = df1.drop(['gender'], axis=1)

    cd_dummies = pd.get_dummies(df1['congressional_district'], prefix='cd', dtype='int64')
    df1[cd_dummies.columns] = cd_dummies
    df1 = df1.drop(['congressional_district'], axis=1)

    df1['age'] = 2020 - df1['birthyear']
    df1['age'] = df1['age'].astype('int64')
    #df1['age'] = np.log(df1['age'])
    df1 = df1.drop(['birthyear'], axis=1)

    counties = ['Appling', 
            'Atkinson',
            'Bacon',
            'Baker',
            'Baldwin',
            'Banks',
            'Barrow',
            'Bartow',
            'Ben_Hill',
            'Berrien',
            'Bibb',
            'Bleckley',
            'Brantley',
            'Brooks',
            'Bryan',
            'Bulloch',
            'Burke',
            'Butts',
            'Calhoun',
            'Camden',
            'Candler',
            'Carroll',
            'Catoosa',
            'Charlton',
            'Chatham',
            'Chattahoochee',
            'Chattooga',
            'Cherokee',
            'Clarke',
            'Clay',
            'Clayton',
            'Clinch',
            'Cobb',
            'Coffee',
            'Colquitt',
            'Columbia',
            'Cook',
            'Coweta',
            'Crawford',
            'Crisp',
            'Dade',
            'Dawson',
            'De_Kalb',
            'Decatur',
            'Dodge',
            'Dooly',
            'Dougherty',
            'Douglas',
            'Early',
            'Echols',
            'Effingham',
            'Elbert',
            'Emanuel',
            'Evans',
            'Fannin',
            'Fayette',
            'Floyd',
            'Forsyth',
            'Franklin',
            'Fulton',
            'Gilmer',
            'Glascock',
            'Glynn',
            'Gordon',
            'Grady',
            'Greene',
            'Gwinnett',
            'Habersham',
            'Hall',
            'Hancock',
            'Haralson',
            'Harris',
            'Hart',
            'Heard',
            'Henry',
            'Houston',
            'Irwin',
            'Jackson',
            'Jasper',
            'Jeff_Davis',
            'Jefferson',
            'Jenkins',
            'Johnson',
            'Jones',
            'Lamar',
            'Lanier',
            'Laurens',
            'Lee',
            'Liberty',
            'Lincoln',
            'Long',
            'Lowndes',
            'Lumpkin',
            'Macon',
            'Madison',
            'Marion',
            'McDuffie',
            'McIntosh',
            'Meriwether',
            'Miller',
            'Mitchell',
            'Monroe',
            'Montgomery',
            'Morgan',
            'Murray',
            'Muscogee',
            'Newton',
            'Oconee',
            'Oglethorpe',
            'Paulding',
            'Peach',
            'Pickens',
            'Pierce',
            'Pike',
            'Polk',
            'Pulaski',
            'Putnam',
            'Quitman',
            'Rabun',
            'Randolph',
            'Richmond',
            'Rockdale',
            'Schley',
            'Screven',
            'Seminole',
            'Spalding',
            'Stephens',
            'Stewart',
            'Sumter',
            'Talbot',
            'Taliaferro',
            'Tattnall',
            'Taylor',
            'Telfair',
            'Terrell',
            'Thomas',
            'Tift',
            'Toombs',
            'Towns',
            'Treutlen',
            'Troup',
            'Turner',
            'Twiggs',
            'Union',
            'Upson',
            'Walker',
            'Walton',
            'Ware',
            'Warren',
            'Washington',
            'Wayne',
            'Webster',
            'Wheeler',
            'White',
            'Whitfield',
            'Wilcox',
            'Wilkes',
            'Wilkinson',
            'Worth'
           ]
    keys = range(1,161)
    county_dict = {}
    for key in keys:
        for county in counties:
            county_dict[key] = county
            counties.remove(county)
        break
    df1['county_code'] = df1['county_code'].replace(county_dict)
    df1 = df1.rename(columns={'county_code': 'county'})

    rural = ['Appling', 
            'Atkinson',
            'Bacon',
            'Baker',
            'Baldwin',
            'Banks',          
            'Ben_Hill',
            'Berrien',
            'Bleckley',
            'Brantley',
            'Brooks',
            'Bryan',
            'Burke',
            'Butts',
            'Calhoun',
            'Candler',
            'Charlton',
            'Chattahoochee',
            'Chattooga',
            'Clay',
            'Clinch',
            'Coffee',
            'Colquitt',
            'Cook',
            'Crawford',
            'Crisp',
            'Dade',
            'Dawson',
            'Decatur',
            'Dodge',
            'Dooly',
            'Early',
            'Echols',
            'Elbert',
            'Emanuel',
            'Evans',
            'Fannin',
            'Franklin',
            'Gilmer',
            'Glascock',
            'Grady',
            'Greene',
            'Habersham',
            'Hancock',
            'Haralson',
            'Harris',
            'Hart',
            'Heard',
            'Irwin',
            'Jasper',
            'Jeff_Davis',
            'Jefferson',
            'Jenkins',
            'Johnson',
            'Jones',
            'Lamar',
            'Lanier',
            'Laurens',
            'Lee',
            'Lincoln',
            'Long',
            'Lumpkin',
            'Macon',
            'Madison',
            'Marion',
            'McDuffie',
            'McIntosh',
            'Meriwether',
            'Miller',
            'Mitchell',
            'Monroe',
            'Montgomery',
            'Morgan',
            'Murray',
            'Oconee',
            'Oglethorpe',
            'Peach',
            'Pickens',
            'Pierce',
            'Pike',
            'Polk',
            'Pulaski',
            'Putnam',
            'Quitman',
            'Rabun',
            'Randolph',
            'Schley',
            'Screven',
            'Seminole',
            'Stephens',
            'Stewart',
            'Sumter',
            'Talbot',
            'Taliaferro',
            'Tattnall',
            'Taylor',
            'Telfair',
            'Terrell',
            'Thomas',
            'Tift',
            'Toombs',
            'Towns',
            'Treutlen',
            'Turner',
            'Twiggs',
            'Union',
            'Upson',
            'Ware',
            'Warren',
            'Washington',
            'Wayne',
            'Webster',
            'Wheeler',
            'White',
            'Wilcox',
            'Wilkes',
            'Wilkinson',
            'Worth'
           ]
    urban = ['Barrow',
            'Bartow',
            'Bibb',
            'Bulloch',
            'Carroll',
            'Catoosa',
            'Chatham',
            'Cherokee',
            'Clarke',
            'Clayton',
            'Cobb',
            'Columbia',
            'Coweta',
            'De_Kalb',
            'Dougherty',
            'Douglas',
            'Effingham',
            'Fayette',
            'Floyd',
            'Forsyth',
            'Fulton',
            'Glynn',
            'Gordon',
            'Gwinnett',
            'Hall',
            'Henry',
            'Houston',
            'Jackson',
            'Lowndes',
            'Muscogee',
            'Newton',
            'Paulding',
            'Richmond',
            'Rockdale',
            'Spalding',
            'Troup',
            'Walker',
            'Walton',
            'Whitfield'
            ]
    military = ['Camden','Liberty']

    r_mask = df1['county'].isin(rural)
    u_mask = df1['county'].isin(urban)
    m_mask = df1['county'].isin(military)

    df1['rural'] = r_mask
    df1['urban'] = u_mask
    df1['military'] = m_mask

    df1['rural'] = df1['rural'].map({True: int(1), False: int(0)})
    df1['urban'] = df1['urban'].map({True: int(1), False: int(0)})
    df1['military'] = df1['military'].map({True: int(1), False: int(0)})

    df1 = df1.drop('county', axis=1)

    return df1

def split_data(df, train_ratio):
    df = df.copy()
    msk = np.random.rand(len(df)) < train_ratio
    train = df[msk]
    test = df[~msk]
            
    return train, test


if __name__=='__main__':
    all = get_data('tbl_prod_GABU202012_all.csv')

    oct_new = get_data('tbl_prod_GABU202010_new_records.csv')
    nov_new = get_data('tbl_prod_GABU202011_new_records.csv')
    dec_new = get_data('tbl_prod_GABU202012_new_records.csv')

    new = pd.concat([oct_new, nov_new, dec_new], axis=0)

    # sample data for GitHub
    all_sample = all.sample(10000)
    new_sample = new.sample(1000)

    all_sample.to_csv('../data/all_sample.csv', index = False)
    print('all_sample df saved to ../data/all_sample.csv')
    new_sample.to_csv('../data/new_sample.csv',index = False)
    print('new_sample df saved to ../data/new_sample.csv')

    # tot = combine_dfs(all,new)

    # train, test = split_data(tot, 0.8)

    # train.to_csv('../data/train.csv', index = False)
    # print('train df saved to ../data/train.csv')
    # test.to_csv('../data/test.csv', index = False)
    # print('test df saved to ../data/test.csv')
    
