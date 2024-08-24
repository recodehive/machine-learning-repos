# Bug Hunter Analysis 

The Dataset used here is taken from the Kaggle database website. You can download the file from the link given here, [Bug Hunter Analysis](https://www.kaggle.com/datasets/vellyy/bug-hunter/data)

## About the dataset

There are 5 types of datasets:

- `full`
- `gcf`
- `remove`
- `single`
- `subtract`

And each directory contains the dataset of all the 15 categories as follows:

- `Android-Universal-Image-Loader`
- `BroadleafCommerce`
- `MapDB`
- `antlr4`
- `ceylon-ide-eclipse`
- `elasticsearch`
- `hazelcast`
- `junit`
- `mcMMO`
- `mct`
- `neo4j`
- `netty`
- `orientdb`
- `oryx`
- `titan`

There is 1 more dataset in each folder with named `all` which contains the combined dataset of all categories.

So, Targetting to `full/all` directory which have all categories type of dataset without going one by one each and every dataset.

### Feature of the dataset are explained below:

| Abbreviation | Full name                                   | Method | Class | File |
|--------------|---------------------------------------------|--------|-------|------|
| CLOC         | Comment Lines of Code                       | X      | X     | X    |
| LOC          | Lines of Code                               | X      | X     | X    |
| LLOC         | Logical Lines of Code                       | X      | X     | X    |
| NL           | Nesting Level                               | X      |       |      |
| NLE          | Nesting Level Else-If                       | X      |       |      |
| NII          | Number of Incoming Invocations              | X      |       |      |
| NOI          | Number of Outgoing Invocations              | X      |       |      |
| CD           | Comment Density                             | X      |       |      |
| DLOC         | Documentation Lines of Code                 | X      |       |      |
| TCD          | Total Comment Density                       | X      |       |      |
| TCLOC        | Total Comment Lines of Code                 | X      |       |      |
| NOS          | Number of Statements                        | X      |       |      |
| TLOC         | Total Lines of Code                         | X      |       |      |
| TLLOC        | Total Logical Lines of Code                 | X      |       |      |
| TNOS         | Total Number of Statements                  | X      |       |      |
| McCC         | McCabe’s Cyclomatic Complexity              | X      | X     |      |
| PDA          | Public Documented API                       |        | X     | X    |
| PUA          | Public Undocumented API                     |        | X     | X    |
| HCPL         | Halstead Calculated Program Length          | X      |       |      |
| HDIF         | Halstead Difficulty                         | X      |       |      |
| HEFF         | Halstead Effort                             | X      |       |      |
| HNDB         | Halstead Number of Delivered Bugs           | X      |       |      |
| HPL          | Halstead Program Length                     | X      |       |      |
| HPV          | Halstead Program Vocabulary                 | X      |       |      |
| HTRP         | Halstead Time Required to Program           | X      |       |      |
| HVOL         | Halstead Volume                             | X      |       |      |
| MIMS         | Maintainability Index (Microsoft version)   | X      |       |      |
| MI           | Maintainability Index (Original version)    | X      |       |      |
| MISEI        | Maintainability Index (SEI version)         | X      |       |      |
| MISM         | Maintainability Index (SourceMeter version) | X      |       |      |
| NUMPAR       | Number of Parameters                        | X      |       |      |
| LCOM5        | Lack of Cohesion in Methods 5               | X      | X     |      |
| WMC          | Weighted Methods per Class                  | X      | X     |      |
| CBO          | Coupling Between Object classes             | X      | X     |      |
| CBOI         | Coupling Between Object classes Inverse     | X      | X     |      |
| RFC          | Response set For Class                      | X      | X     |      |
| AD           | API Documentation                           |        | X     |      |
| DIT          | Depth of Inheritance Tree                   | X      | X     |      |
| NOA          | Number of Ancestors                         | X      | X     |      |
| NOC          | Number of Children                          | X      | X     |      |
| NOD          | Number of Descendants                       | X      | X     |      |
| NOP          | Number of Parents                           | X      | X     |      |
| NA           | Number of Attributes                        | X      | X     |      |
| NG           | Number of Getters                           | X      | X     |      |
| NLA          | Number of Local Attributes                  | X      | X     |      |
| NLG          | Number of Local Getters                     | X      | X     |      |
| NLM          | Number of Local Methods                     | X      | X     |      |
| NLPA         | Number of Local Public Attributes           | X      | X     |      |
| NLPM         | Number of Local Public Methods              | X      | X     |      |
| NLS          | Number of Local Setters                     | X      | X     |      |
| NM           | Number of Methods                           | X      | X     |      |
| NPA          | Number of Public Attributes                 | X      | X     |      |
| NPM          | Number of Public Methods                    | X      | X     |      |
| NS           | Number of Setters                           | X      | X     |      |
| TNA          | Total Number of Attributes                  | X      | X     |      |
| TNG          | Total Number of Getters                     | X      | X     |      |
| TNLA         | Total Number of Local Attributes            | X      | X     |      |
| TNLG         | Total Number of Local Getters               | X      | X     |      |
| TNLM         | Total Number of Local Methods               | X      | X     |      |
| TNLPA        | Total Number of Local Public Attributes     | X      | X     |      |
| TNLPM        | Total Number of Local Public Methods        | X      | X     |      |
| TNLS         | Total Number of Local Setters               | X      | X     |      |
| TNM          | Total Number of Methods                     | X      | X     |      |
| TNPA         | Total Number of Public Attributes           | X      | X     |      |
| TNPM         | Total Number of Public Methods              | X      | X     |      |
| TNS          | Total Number of Setters                     | X      | X     |      |

For getting visual represantation go through main readme file or notebook.
