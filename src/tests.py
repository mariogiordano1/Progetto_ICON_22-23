import pandas as pd
df_completo = pd.read_csv("Database/Dataset_Completato_PLS.csv")
db_cities = df_completo["Country"]
db_cities.to_csv("Database/Countries.txt", index=False)
infile = "Database/Countries.txt"
outfilename = "Database/Countries_No_Duplicate.txt"
lines_seen = set() # holds lines already seen
outfile = open(outfilename, "w")
for line in open(infile, "r"):
    if line not in lines_seen: # not a duplicate
        lines_seen.add(line)
outfile.writelines(sorted(lines_seen))
outfile.close()