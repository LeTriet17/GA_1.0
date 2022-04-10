from ga import *

resource_path = "resource.csv"
resource_data = pd.read_csv(resource_path)
# resource_data.columns = resource_data.columns.str.lower()
# resource_data = resource_data[['date', 'manday_ht', 'manday_mt', 'bdpocdiscipline']]
# resource_data = resource_data.rename(columns={"manday_ht": "HT", "manday_mt": "MT"})
# resource_data.date = resource_data.date.apply(lambda row: row[:-4] + "000" + row[-1])
resource_data = resource_data.loc[resource_data['bdpocdiscipline'] == 'E&I']
# print(resource_data)
date_unique = np.unique(resource_data.date.to_list()).astype(list)
# print(date_unique)

def get_resource(team, date, site):  # return side resource
    if date not in date_unique:
        return -1
    return resource_data[(resource_data['bdpocdiscipline'] == team) & (resource_data['date'] == date)][site].item()


# Preprocessing WONUM_data
data_path = "data.csv"
data = pd.read_csv(data_path)


data = data.dropna()
data = data.drop('Unnamed: 0', axis=1)
data = data[data.site != 'Not Defined']
data = data.loc[data['bdpocdiscipline'] == 'E&I']
data = data.reset_index()
data = data.drop('index', axis=1)
dict_wonum = {x: y for x, y in zip(data.wonum, data.index)}

def debug_partial(chromosome):  # fitness function
    MANDAY = dict()
    HC_score = 0
    # SC_score = 0
    # violate_child = dict()

    for child in chromosome:
        # print(chromosome)
        # take bit and convert to component
        component = child.split(
            '-')  # convert: H13807098-01/12/0001-09/03/0002-10110000110 to ['H13807098', '01/12/0001', '09/03/0002', '10110000110']
        # take component
        wonum = component[0]
        target_date_begin = component[1]
        end_date_begin = component[2]
        bit_date = component[3]

        date_begin = decode_datetime(bit_date).split('/')
        date = int(date_begin[0])
        month = int(date_begin[1])
        year = int(date_begin[2])
        # if month > 12 or month < 1 or date > 31 or year > 2:
        #     # SC_score += 1
        #     HC_score += 1
        #     continue
        # if (month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12) and date > 31:
        #     # SC_score += 1
        #     HC_score += 1
        #     continue
        # if month == 2 and date > 28:
        #     HC_score += 1
        #     continue
        # if (month == 4 or month == 6 or month == 9 or month == 11) and date > 30:
        #     # SC_score += 1
        #     HC_score += 1
        #     continue
        date_begin = decode_datetime(bit_date)
        # access from dataframe
        est_dur = access_row_by_wonum(wonum)['r_estdur']
        site = access_row_by_wonum(wonum)['site']
        team = access_row_by_wonum(wonum)['bdpocdiscipline']

        # convert to datetime type
        dt_begin = datetime.datetime.strptime(date_begin, '%d/%m/%Y')
        dt_end = datetime.datetime.strptime(date_begin, '%d/%m/%Y') + datetime.timedelta(days=int(est_dur))
        std_begin = datetime.datetime.strptime(target_date_begin, '%d/%m/%Y')  # start target_day datetime
        etd_end = datetime.datetime.strptime(end_date_begin, '%d/%m/%Y')  # end target_day datetime

        # duration_start = (std_begin - dt_begin).days
        # duration_end = (dt_end - etd_end).days

        # #compute violate point in every element
        # if point_duration(duration_start):
        #     HC_score += 1
        #     continue
        # if point_duration(duration_end):
        #     HC_score += 1
        #     continue
        # violate_child[wonum] = point
        print("\n==========\n")
        print(est_dur)
        print("\n==========\n")
        tup = (team, convert_datetime_to_string(dt_begin), site)

        MANDAY[tup] = MANDAY.get(tup, 0) + 1
        # compute manday resource
        for i in range(1, est_dur):
            run_date = dt_begin + datetime.timedelta(days=i)
            tup_temp = (team, convert_datetime_to_string(run_date), site)

            MANDAY[tup_temp] = MANDAY.get(tup_temp, 0) + 1

        # tup = (team, convert_datetime_to_string(dt_end), site)
        # MANDAY[tup] = MANDAY.get(tup, 0) + 1
    # print violate_child
    
    for key, value in MANDAY.items():
        team, date, site = key
        data_resource_value = get_resource(team, date, site)
        print(data_resource_value)
        # if data_resource_value == -1:  # gen date with date not in resouce
        #     HC_score += 1
        # elif data_resource_value < value:
        #     HC_score += 1
        print(team,date,site,value)
    return HC_score
    # ,SC_score
    
def beautiful_print(chromosome):
    for i,line in enumerate(chromosome):
        temp_line = str(line).split('-')
        temp_line[-1] = decode_datetime(temp_line[-1])
        temp_line = '-'.join(temp_line)
        chromosome[i] = temp_line
    return chromosome


np.random.seed(0)
# chromosome = ['H13831665-06/03/0002-31/03/0002-01111001110',
#  'H13831669-06/03/0002-31/03/0002-11101001110',
#  'H13831673-06/03/0002-31/03/0002-01001001110',
#  'H13831677-06/03/0002-31/03/0002-01011001110',
#  'H13831681-06/03/0002-31/03/0002-11000001110',
#  'H13831685-06/03/0002-31/03/0002-10101001110',
#  'H13831693-06/03/0002-31/03/0002-10001001110',
#  'H13831697-06/03/0002-31/03/0002-01001001110',
#  'H13831701-06/03/0002-31/03/0002-10110001110',
#  'H13831709-06/03/0002-31/03/0002-01100001110',
#  'H13831713-06/03/0002-31/03/0002-10100001110',
#  'H13831717-06/03/0002-31/03/0002-10100001110']

chromosome = createParent(1)[0]
print(len(chromosome))
# print(debug_partial(chromosome))
# chromosome = beautiful_print(chromosome)
# for element in chromosome:
#     print(element)
