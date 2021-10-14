# Data-Analysis
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<center>\n",
    "    <img src=\"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/images/IDSNlogo.png\" width=\"300\" alt=\"cognitiveclass.ai logo\"  />\n",
    "</center>\n",
    "\n",
    "<h1 align=\"center\"><font size=\"5\">Classification with Python</font></h1>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "In this notebook we try to practice all the classification algorithms that we have learned in this course.\n",
    "\n",
    "We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.\n",
    "\n",
    "Let's first load required libraries:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "import itertools\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.ticker import NullFormatter\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.ticker as ticker\n",
    "from sklearn import preprocessing\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### About dataset\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:\n",
    "\n",
    "| Field          | Description                                                                           |\n",
    "|----------------|---------------------------------------------------------------------------------------|\n",
    "| Loan_status    | Whether a loan is paid off on in collection                                           |\n",
    "| Principal      | Basic principal loan amount at the                                                    |\n",
    "| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |\n",
    "| Effective_date | When the loan got originated and took effects                                         |\n",
    "| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |\n",
    "| Age            | Age of applicant                                                                      |\n",
    "| Education      | Education of applicant                                                                |\n",
    "| Gender         | The gender of applicant                                                               |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let's download the dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2021-08-16 00:12:50--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv\n",
      "Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 198.23.119.245\n",
      "Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|198.23.119.245|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 23101 (23K) [text/csv]\n",
      "Saving to: ‘loan_train.csv’\n",
      "\n",
      "loan_train.csv      100%[===================>]  22.56K  --.-KB/s    in 0s      \n",
      "\n",
      "2021-08-16 00:12:50 (234 MB/s) - ‘loan_train.csv’ saved [23101/23101]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Load Data From CSV File\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>9/22/2016</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>10/8/2016</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>10/8/2016</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30       9/8/2016   \n",
       "1           2             2     PAIDOFF       1000     30       9/8/2016   \n",
       "2           3             3     PAIDOFF       1000     15       9/8/2016   \n",
       "3           4             4     PAIDOFF       1000     30       9/9/2016   \n",
       "4           6             6     PAIDOFF       1000     30       9/9/2016   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0  10/7/2016   45  High School or Below    male  \n",
       "1  10/7/2016   33              Bechalor  female  \n",
       "2  9/22/2016   27               college    male  \n",
       "3  10/8/2016   28               college  female  \n",
       "4  10/8/2016   29               college    male  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('loan_train.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(346, 10)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Convert to date time object\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0 2016-10-07   45  High School or Below    male  \n",
       "1 2016-10-07   33              Bechalor  female  \n",
       "2 2016-09-22   27               college    male  \n",
       "3 2016-10-08   28               college  female  \n",
       "4 2016-10-08   29               college    male  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['due_date'] = pd.to_datetime(df['due_date'])\n",
    "df['effective_date'] = pd.to_datetime(df['effective_date'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Data visualization and pre-processing\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let’s see how many of each class is in our data set\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PAIDOFF       260\n",
       "COLLECTION     86\n",
       "Name: loan_status, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['loan_status'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "260 people have paid off the loan on time while 86 have gone into collection\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's plot some columns to underestand data better:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting package metadata (current_repodata.json): done\n",
      "Solving environment: done\n",
      "\n",
      "# All requested packages already installed.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# notice: installing seaborn might takes a few minutes\n",
    "!conda install -c anaconda seaborn -y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'Principal', bins=bins, ec=\"k\")\n",
    "\n",
    "g.axes[-1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "bins = np.linspace(df.age.min(), df.age.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'age', bins=bins, ec=\"k\")\n",
    "\n",
    "g.axes[-1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Pre-processing:  Feature selection/extraction\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Let's look at the day of the week people get the loan\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['dayofweek'] = df['effective_date'].dt.dayofweek\n",
    "bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'dayofweek', bins=bins, ec=\"k\")\n",
    "g.axes[-1].legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "      <th>weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  dayofweek  weekend  \n",
       "0 2016-10-07   45  High School or Below    male          3        0  \n",
       "1 2016-10-07   33              Bechalor  female          3        0  \n",
       "2 2016-09-22   27               college    male          3        0  \n",
       "3 2016-10-08   28               college  female          4        1  \n",
       "4 2016-10-08   29               college    male          4        1  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Convert Categorical features to numerical values\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let's look at gender:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender  loan_status\n",
       "female  PAIDOFF        0.865385\n",
       "        COLLECTION     0.134615\n",
       "male    PAIDOFF        0.731293\n",
       "        COLLECTION     0.268707\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "86 % of female pay there loans while only 73 % of males pay there loan\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let's convert male to 0 and female to 1:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "      <th>weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  dayofweek  weekend  \n",
       "0 2016-10-07   45  High School or Below       0          3        0  \n",
       "1 2016-10-07   33              Bechalor       1          3        0  \n",
       "2 2016-09-22   27               college       0          3        0  \n",
       "3 2016-10-08   28               college       1          4        1  \n",
       "4 2016-10-08   29               college       0          4        1  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## One Hot Encoding\n",
    "\n",
    "#### How about education?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "education             loan_status\n",
       "Bechalor              PAIDOFF        0.750000\n",
       "                      COLLECTION     0.250000\n",
       "High School or Below  PAIDOFF        0.741722\n",
       "                      COLLECTION     0.258278\n",
       "Master or Above       COLLECTION     0.500000\n",
       "                      PAIDOFF        0.500000\n",
       "college               PAIDOFF        0.765101\n",
       "                      COLLECTION     0.234899\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(['education'])['loan_status'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Features before One Hot Encoding\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>education</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>High School or Below</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>Bechalor</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender             education\n",
       "0       1000     30   45       0  High School or Below\n",
       "1       1000     30   33       1              Bechalor\n",
       "2       1000     15   27       0               college\n",
       "3       1000     30   28       1               college\n",
       "4       1000     30   29       0               college"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Principal','terms','age','Gender','education']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Use one hot encoding technique to convert categorical varables to binary variables and append them to the feature Data Frame\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   45       0        0         0                     1   \n",
       "1       1000     30   33       1        0         1                     0   \n",
       "2       1000     15   27       0        0         0                     0   \n",
       "3       1000     30   28       1        1         0                     0   \n",
       "4       1000     30   29       0        1         0                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Feature = df[['Principal','terms','age','Gender','weekend']]\n",
    "Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)\n",
    "Feature.drop(['Master or Above'], axis = 1,inplace=True)\n",
    "Feature.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Feature Selection\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let's define feature sets, X:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   45       0        0         0                     1   \n",
       "1       1000     30   33       1        0         1                     0   \n",
       "2       1000     15   27       0        0         0                     0   \n",
       "3       1000     30   28       1        1         0                     0   \n",
       "4       1000     30   29       0        1         0                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = Feature\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "What are our lables?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df['loan_status'].values\n",
    "y[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Normalize Data\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Data Standardization give data zero mean and unit variance (technically should be done after train test split)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.51578458,  0.92071769,  2.33152555, -0.42056004, -1.20577805,\n",
       "        -0.38170062,  1.13639374, -0.86968108],\n",
       "       [ 0.51578458,  0.92071769,  0.34170148,  2.37778177, -1.20577805,\n",
       "         2.61985426, -0.87997669, -0.86968108],\n",
       "       [ 0.51578458, -0.95911111, -0.65321055, -0.42056004, -1.20577805,\n",
       "        -0.38170062, -0.87997669,  1.14984679],\n",
       "       [ 0.51578458,  0.92071769, -0.48739188,  2.37778177,  0.82934003,\n",
       "        -0.38170062, -0.87997669,  1.14984679],\n",
       "       [ 0.51578458,  0.92071769, -0.3215732 , -0.42056004,  0.82934003,\n",
       "        -0.38170062, -0.87997669,  1.14984679]])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X= preprocessing.StandardScaler().fit(X).transform(X)\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Classification\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model\n",
    "You should use the following algorithm:\n",
    "\n",
    "*   K Nearest Neighbor(KNN)\n",
    "*   Decision Tree\n",
    "*   Support Vector Machine\n",
    "*   Logistic Regression\n",
    "\n",
    "\\__ Notice:\\__\n",
    "\n",
    "*   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.\n",
    "*   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.\n",
    "*   You should include the code of the algorithm in the following cells.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# K Nearest Neighbor(KNN)\n",
    "\n",
    "Notice: You should find the best k to build the model with the best accuracy.\\\n",
    "**warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set: (242, 8) (242,)\n",
      "Test set: (104, 8) (104,)\n"
     ]
    }
   ],
   "source": [
    "#Train/Test Split\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)\n",
    "print ('Train set:', X_train.shape,  y_train.shape)\n",
    "print ('Test set:', X_test.shape,  y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAagAAAEYCAYAAAAJeGK1AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABIFklEQVR4nO3dd3iUVfbA8e+ZmUwqHQQFVFSa0qUoCCo2LNhABQsuKi4K1rWgq6y69v2JrooFUdBVQAURbIiKiAoqLdKRXkRqKGmTaff3xyQYIWUy876ZmeR8nocHMnnnvichmTO3nSvGGJRSSql444h1AEoppVRJNEEppZSKS5qglFJKxSVNUEoppeKSJiillFJxSROUUkqpuGRrghKRPiKyWkTWisiIEj5fS0Q+EZFfRWS5iAwu9rmNIrJURDJFZIGdcSqllIo/Ytc+KBFxAr8B5wBbgfnAQGPMimLXPAjUMsbcLyINgNVAI2OMV0Q2Ap2NMbttCVAppVRcs7MH1RVYa4xZb4zxApOASw65xgA1RESADCAL8NsYk1JKqQThsrHtxsCWYh9vBbodcs3LwHRgG1ADuMoYEyz8nAFmiogBXjfGjCnpJiJyM3AzQHp6+smtWrWy7itQSillu4ULF+42xjQ49HE7E5SU8Nih44nnAZlAb+B44CsR+d4YcwDoYYzZJiJHFD6+yhgz57AGQ4lrDEDnzp3NggU6XaWUUolERDaV9LidQ3xbgabFPm5CqKdU3GDgIxOyFtgAtAIwxmwr/HsnMJXQkKFSSqlqws4ENR9oLiLNRMQNDCA0nFfcZuAsABFpCLQE1otIuojUKHw8HTgXWGZjrEoppeKMbUN8xhi/iAwHvgScwFvGmOUiMrTw868B/wbGi8hSQkOC9xtjdovIccDU0NoJXMAEY8wMu2JVSikVf2xbZh4LOgellLKSz+dj69ateDyeWIdSJaSkpNCkSROSkpL+8riILDTGdD70ejsXSSilVELbunUrNWrU4Nhjj6VwREdFyBjDnj172Lp1K82aNQvrOVrqSCmlSuHxeKhXr54mJwuICPXq1atQb1QTlFJKlUGTk3Uq+r3UBKWUUnFu6tSpiAirVq2KdSiVShOUUnEuEDR4/cHyL1RV1sSJEznttNOYNGmSrfcJBAK2tl9RmqCUinM5Hj8HPD6q0opbFb6cnBx+/PFH3nzzzb8kqEAgwD333EPbtm1p164dL730EgDz58+ne/futG/fnq5du5Kdnc348eMZPnz4wededNFFzJ49G4CMjAxGjhxJt27dmDdvHo899hhdunShTZs23HzzzQd/7tauXcvZZ59N+/bt6dSpE+vWreO6665j2rRpB9u95pprmD790O2ukdNVfErFMa8/iMcfelebU+CnRkpSOc9Qdrlzxp1kbs+0tM0OjTrwQp8Xyrzm448/pk+fPrRo0YK6deuyaNEiOnXqxJgxY9iwYQOLFy/G5XKRlZWF1+vlqquu4v3336dLly4cOHCA1NTUMtvPzc2lTZs2PPbYYwCceOKJjBw5EoDrrruOTz/9lL59+3LNNdcwYsQILrvsMjweD8FgkJtuuonnn3+eSy65hP379zN37lzefvttS743oD0opeJatsd38N953oAO9VVDEydOZMCAAQAMGDCAiRMnAvD1118zdOhQXK5QP6Nu3bqsXr2aI488ki5dugBQs2bNg58vjdPppF+/fgc//vbbb+nWrRtt27Zl1qxZLF++nOzsbH7//Xcuu+wyILSfKS0tjdNPP521a9eyc+dOJk6cSL9+/cq9X0VoD0qpOJXn9eMP/nVY74DHR710t64si4Hyejp22LNnD7NmzWLZsmWICIFAABHh2WefxRhz2M9BSY8BuFwugsE/39wUX+qdkpKC0+k8+Pitt97KggULaNq0KY888ggej6fM4eXrrruO9957j0mTJvHWW29F+yX/hfaglIpDwaAhp+Dwo9ECQUOuN74mspV9Jk+ezKBBg9i0aRMbN25ky5YtNGvWjB9++IFzzz2X1157Db8/9HOSlZVFq1at2LZtG/PnzwcgOzsbv9/PscceS2ZmJsFgkC1btvDLL7+UeL+ixFW/fn1ycnKYPHkyEOqJNWnShI8//hiAgoIC8vLyAPjb3/7GCy+8AMBJJ51k6devCUqpOJTj9VPam9a8Aj++gA71VQcTJ048OKxWpF+/fkyYMIGbbrqJo48+mnbt2tG+fXsmTJiA2+3m/fff57bbbqN9+/acc845eDweevToQbNmzWjbti333HMPnTp1KvF+tWvXZsiQIbRt25ZLL7304FAhwP/+9z9efPFF2rVrR/fu3dm+fTsADRs2pHXr1gwePNjyr19r8SkVZ3yBIFm53jKvcTmEehnJlRRR9bVy5Upat24d6zDiWl5eHm3btmXRokXUqlWr3OtL+p6WVotPe1BKxZlsz+FDe4fyBw25JQwBKlWZvv76a1q1asVtt90WVnKqKF0koVQc8fgCYQ/f5Rb4SXY5cDn1faaKjbPPPpvNmzfb1r7+ZCsVJ4wxYfWeDl4PHKjA9UolGk1QSsWJnAI/wQrOCfsCQfK8mqRU1aQJSqk44A8EyY9w+XiOx08gWHUWOylVRBOUUnEgp8BPpCnGAAfyfeVep1Si0QSlVIwV+AMURFnCyBtFD6y6yfb4yPcGCFahXue8efMYMmRImde89tprtG3blg4dOnDaaaexYsWKCt9n3759vPLKK6V+/m9/+9vBzb1W0FV8SsVQRRdGlCXb48PtcuB0aBmk0mR7fOR5A0AombudDlKSnCS7HDjC+L7tOBD+abDhaFgzJexrZ8+ezfjx4xk/fvxhn5sxYwZ9+vQp8/lXX301Q4cOBWD69OncfffdzJgxo0LxFiWoW2+9tULPi5T2oJSKoTxvwLL5I8Nfi8uqv8r3BgqT05+8gSAHPD525xSwL8+bsD2rb775hrPPPrvMa2rWrHnw37m5uQdr9k2dOpWzzz4bYwx//PEHLVq0YPv27SxfvpyuXbvSoUMH2rVrx5o1axgxYgTr1q2jQ4cO3HvvvRhjGD58OCeeeCIXXnghO3futPTr0h6UUjESsGGzbYE/iMcXICXJaWm7ic7rD5aZvA2h712BP4gAbpcjYb6Hu3fvJikpKayNsqNHj2bUqFF4vV5mzZoFwGWXXcaUKVMYPXo0M2bM4NFHH6VRo0Y88cQT3HHHHVxzzTV4vV4CgQBPP/00y5YtIzMzE4CPPvqI1atXs3TpUnbs2MGJJ57IDTfcYNnXpglKqRjJ8US+MKIsBzw+3M7whqyqg0DQsC/fG/b3uniyCgQN/kAQhwixKiDfrVs3CgoKyMnJISsriw4dOgDwzDPPcN555zFz5kzOPffcsNoaNmwYw4YNY8KECTz++OMHz2566aWXaNOmDaeccgoDBw4E4NRTT+WJJ55g69atXH755TRv3vyw9ubMmcPAgQNxOp0cddRR9O7d25ovupAO8SkVA8UPIrSaMeGVS6oOjDHszfOWWng3rDaAgDH4C5NVMGgq9XTjn3/+mczMTMaOHcvFF19MZmYmmZmZnHfeeQB88cUXB+efBg8eTIcOHbjgggvKbHPAgAEHK5MD/P777zgcDnbs2HHwWI6rr76a6dOnk5qaynnnnXewx3UoO49+0QSlVAzYPVfk8Qfw+HRV3748n6V7xGKdrA6LxxiWLFlysFc1btw4MjMz+fzzzw+7ds2aNQf//dlnnx3sEfn9fgYPHsyECRNo3bo1o0aNAmD9+vUcd9xx3H777Vx88cUsWbKEGjVqkJ2dfbCdXr16MWnSJAKBAH/88QfffvutpV+fDvEpVclKOojQDtV9qO+Ax4fXxmNJipIVBgRzcBiwMg+TXLhwIR07dgzrni+//DJff/01SUlJ1KlT5+Dw3pNPPknPnj3p2bMnHTp0oEuXLlx44YV8/PHHvPvuuyQlJdGoUSNGjhxJ3bp16dGjB23atOH888/n2WefZdasWbRt25YWLVpw+umnW/r16XEbSlWiYNCwO7cgqiGnikhxOamVllQ5N4sjeV6/JcOcu7asp2WrVhV6jgihZIX9yerxxx/nhBNOOHgkfCKoyHEb2oNSqhKVdRChHTz+AMm+xFmRZoUCfyCmc3DGFPasAJFQz8phU6J66KGHbGk3XugclFKVxBejag/ZHn9M50kqkz8QZH8clX0yJrSK0BcIEggGK1wMuLrTBKVUJYnVu/qgMdXiWI5g0LAv31epPdSKCGqyqjBbE5SI9BGR1SKyVkRGlPD5WiLyiYj8KiLLRWRwuM9VKpFU5CBCu+5fYNOy9nixP9/aFXtF7Oh9VtdkVdHvpW0JSkScwGjgfOBEYKCInHjIZcOAFcaY9sAZwHMi4g7zuUolBCvr7UXjQH7VHerbn2/Pij2XO5msPVm2ft+qS7IyxrBnzx5SUsKvP2jnIomuwFpjzHoAEZkEXAIUL6FrgBoSWuqSAWQBfqBbGM9VKiFEchChHYLGkF3gp2ZK1VrVl+f127bnq2a9huzbs4Pdu3fZ0n5ZEmJzQOGKxXClpKTQpEmTsK+3M0E1BrYU+3grocRT3MvAdGAbUAO4yhgTFJFwnqtU3IvmIEI75HsDpLicuF1VY/rZ7hV7TpeLOg0b29Z+ohOBI2qE3yOqKDt/SktKq4e+jTwPyASOAjoAL4tIzTCfG7qJyM0iskBEFuzaVfnvcpQqSzQHERb5ct2njF1c+hk8FXXA46sSQ33+QJD9efGzYk9Zz84e1FagabGPmxDqKRU3GHjahH5b1orIBqBVmM8FwBgzBhgDoY261oSuVPSsOIhwd94uhs+4kWzvAVrXb0OPpr2ijisQNOQU+KmRwEN9waBhb57PlmK7Kn7Y2YOaDzQXkWYi4gYGEBrOK24zcBaAiDQEWgLrw3yuUnHLqoURo35+kjxfLo3Sj2TErNvxBrwWRBc6hyqWqwqjYUxoOXk8zOspe9mWoIwxfmA48CWwEvjAGLNcRIaKyNDCy/4NdBeRpcA3wP3GmN2lPdeuWJWymhUHEa7N+o13lozl2rY38n/nvMKarNW8uvAFawIEDuQn5lDfAY8/YZOrqhitxaeUxQJBw56cgqiHnwZPv5Lvt8xm3uBlNEg7ghs/Gcg3G2bw3aBFHFO7mRWhkp7sIiM5cSqe5Rb4ybH4kEcVOasWSZRWi69qLOVRKo5YcRDhvK0/8MW6T7ityz00SDsCgH+f8R+cDhcPfHuXZT2fvAI//gTpjXh8AU1O1YwmKKUsZMVBhEET5NE5IzgqozFDOg4/+PhRNZpwX/eHmbXxSz5b+3GUkYYYSIgySL5AkANxVGNPVQ5NUEpZyIqDCKet/pDMHQsZ0eNR0pLS/vK5GzvcykkN2vHw7HvI8WaX0kLF+AJBcuO4ZxIIGvbpir1qSROUUhax4iBCj9/DEz+OpE2D9vRvPfCwz7scLv5z1ktsz/mD/8z7d1T3Ki43Tof6jDHsy/Pqir1qShOUUhYIFu4titabma+w9cBmHun1NA4p+dez05Fdua7tjbyxeDRLd2ZGfU+I36G+A/mVc/qwik+aoJSygBUHEe7J381/f3mWs5udz2lHn1HmtQ+e9hh1Uupx3ze3EQhaU0rJFwiS542fJJVT4I96Pk8lNk1QSkXJqoMIn//pKXK82Tzc84lyr62dUodHT3+GxdsX8O7St6K+d5Ecj9+WIysqyuMLxPW8mKocmqCUipIVFSPW713L+CVjuLbNDbSs1zqs5/RrNYAeTU/nyR9Hsit3R9QxQOFQX4xXy3n9umJPhWiCUioKVh1E+MQPD5HsTOGeU/8Z9nNEhKd7/5c8Xy6Pznkg6hiKeGNYgT0QNOzL9+qKPQVoglIqYlbV2/v59x/5bO00hnf5B0ekN6rQc5vXbcnwLv9g8qqJ/LB5dtSxFMkusOd02rIUrdjTBXuqiCYopSJkxUGExhgemTOCRulH8vdOt0fUxu1d7+OYWs24f9btFPgLoornz7is2dNVEfvzfbav2NuVu4MRs+7k49UfkOfLs/VeKnqaoJSKgFUHEU77bTKLty8ocVNuuFJdqTzV+wXW7V3DKwufjzqmIgX+oG0n1R4q2+OL+miS8ngDXm769GrG//o6Qz+/nnZjjuWOL2/m+83fWrYSUllLE5RSEbDiIMICfwFP/PAwJzVoxxWtr46qrd7Hnkvf5pfzws9Ps2Hfuigj+9MBj4+gzb2afG+AvEqY83p0zgh+3jaX0X3eYnL/GfRtfjmfr53GFVMuoPObLXhszoOs2LXU9jhU+DRBKVVBVhxECPDWr6+y5cAm/tXzKZwOZ9TtPXbGsyQ53Tw4y7pisqGhPvuWe3v9wUoZSvxwxQTezHyVv3e6jX6tB3Ja09N5/tzXWPL3jYy58F3aHtGRMYtfove7XTnzf10YvWAU27K32h6XKpset6FsEQgaHBJaaVaVGGPYk+uNegFBVv4eTh3XhpOP7MaEyz62Jjhg7OLRPDT7HsZc+C4Xt+hnWbu1UpNISYo+iRYXCBr25BbYvihiyY7FXPx+bzod2ZUP+n2Gy1Hy8SJ78ncz/bcpTF45gYV//IIg9Gh6Ov1bX82FJ1xCjeSa9gaagOw+bkMTlLLF3lwv/qAhI9lFqtvaF7ZYsuo8opGz72Vs5it8c+0vtK5/kgWRhfiDfi6Y2IsduX/ww/W/Wvai6hChXrobh8OaNxzBoCErL/pEX549+bs5b0IPgsEAX14z9+DRJeXZsG8dU1ZOYvLKCWzcv54UZwrnHX8R/VtfzRnHnE2SM8nWuBOFngelEo4vEMQbCBI0hgMeH7tzCiptst1OgaCxpLrBhn3rGPfr61x90t8sTU4QKib7zFkvsjN3B8/MfdSydoPGkG1hZYf9+fYvY/cH/dzy+fXsyt3Bm30nhZ2cAJrVPp57Tv0n8wYv47MBsxnY5nq+3/wt1027nPZvHMeD397Fou3zE/JE4kSiCUpZ7tAX8UDQsD/fx54ET1RWHEQI8MQPD5PkdHPvqQ9Z0NrhOjbqzPXtb+atX19jyY7FlrXr8QUosKA23gGPD28lVE5/6sd/MWfzLJ7q/V86NjrszXlYRISTj+zGU71fIPPm9bxz8WROa3o67y0dxwUTe9FjfDue++lJNu3bYHH0CnSIT1nMHwiyJ9db5jUuh5CR4iLZlThDf15/kL15ZX9d4Viw7Scuev9M7jnloQpVjaio/Z59nPZ2BxrXaMJnA76zZBEGhIb66me4I55bzPP6bV10UWT6b1O4+bNrGdRuCM+e9aLl7R8o2M+naz5myqqJzN0yB4Ohy5Gn0K/1QC5u0Y+6qfUsv2c80jmoCtAEFXv783xhV6BOcjpIT3YmRKLak1MQ9SZSYwx93z+TLQc2MXfwMtKT0i2KrmRTV73PLV/8jad6v8Dg9n+3rN1Ut5OaKRWfgynwB9iXZ/+KvVW7V3DBpF60rn8SH/WfSbIr2db7/Z69hY9Wvc/klRNZvWcFSY4kzmp2Hv1aDeSc4y4gxRX9C3i80gRVAZqgYiuc3lNJ3E4H6cku3K74HHHO9wY4YMFS6E9++4ghn13DqHNe5eo2f4s+sHIYY7hyyoVk7ljIj3/7tcJllMpSJ81dof8vfyBIViWUMdrv2cf5E3uS7c3mq2vm0ijjKHtvWIwxhhW7l/LhyglMXfU+O3K3UzO5Fn2bX06/1gM5pXGPUs/4SlSaoCpAE1Rs7c/3RTXHlOwKJaokZ/z8EgeDht0WLIX2Brz0fLsDqa40vrn2Z8uG3Mqzbu8azvxfZy484VJeveBty9p1OkKr+sIZ6qusFXtBE+T6aVfw7aaZTOk/g26Ne9h6v7IEggF+3PIdk1dN5LM1H5Pry6Fxjab0azWAfq0Hhl2xPt7pKj6VEAJBQ0GUCyAK/EGycr3sy/PGzfHjVhxECDD+19fZtH8D/+plzabccB1fpzm3dbmXqas/YM6mWZa1GwjzBGFjTKWs2AMY9dNTfLXhcx49/dmYJicAp8NJr2N68+J5b7Dk7xt59fzxtKp/EqMXjOL0dzpx7nvdeX3Ri+zI+SOmccY77UEpSxzw+Cw/oiHF5SQ92YkrRj0qXyCUMKO1z7OXU946iQ6NTmbS5Z9YEFnFePwezvxfZwRh1nXzLZ0TqZvuLrPHG22vOlwz13/OoGn9uKL1Nbx43htxu0F8V+4Opv02mckrJ5K5YyEOcdDr6N70b3015x/fl3R3RqxDrBDtQam4FwwaPDbUUvP4A+zJ9VbaO/BDWbXa7IVfnmF/wT5G9nzSkvYqKsWVwtO9/8v6fWt5ef5zlra9P99X6l6g3AJ/pSSn9XvXMnzGDbQ9ogPPnv1S3CYngAbpDbmp4zBmXP0D31+fyR1d72Pd3jUMn3EDbV4/hmFfDObbjV/hD+ppwqA9KGWBbI/P9mKfAqS4naS7XTgtqmZQFo8vwH4LTnXdtG8DPd/pQL9WA3n+3NcsiCxyQz8fxOdrpzH7uoUcV+cEy9pNT3aRkfzX8kFWff/Kk+vN4YJJp7Mrdwczrv6Ro2sdY/s9rWaM4Zdtc5mychLTf5vCvoK9NEhryOlH9ybJ6Y51eOXqeUw3bulyS1RtlNaDKrkolVJhCgYN+ZXwLtkQWk3n8QZILUxUVpXdOexeFh1ECPDkjyNxiov7uj9sSXvReLTXM3yz4UsemHUnky7/xLKeRl6Bn2SX4+BQnz9QOUe2G2O466uhrMlaxaTLPknI5AShzcDdGvegW+Me/PuM/2PWxi+ZvHIC837/PiEqVdRMsW8ZvSYoFZV8X6BST0A1QJ43QL6NiSrXG4j6IEKAhX/8zLTfJnN3twc5MqNxxO24nQ5cTom6l9ow40ge6PEoD357F9N++5BLW14ZVXtFDHAg30e9jGSCQcPePF+lHNn+ysLnmf7bFB467XF6HdO7Eu5ov2RXMuefcDHnn3BxrEMJi1VzUKXROSgVMWMMud7YjJUXJarduQWhs5ksypKBoCHPgppzxhgenfMADdIaMqzzXVG1VTSE5rCgx3N9uyG0b9iJkbPvZ79nX9TtFfEXrurbl++zJLmXZ86mWTzxw8P0bX45wzrfbfv9VGxoglIRq+zeU0mMCU3G78opINeCRJXtsebd/+drp/HLtnnc331kVCuzkl0O3C4HInLYPE8knA4nz571Ervzd/L03Eeibq+43AI/vkrYHrDlwCaGfj6IE+q05IVzX4/rRREqOrYmKBHpIyKrRWStiIwo4fP3ikhm4Z9lIhIQkbqFn9soIksLP6crH+KMMYbcgvgp/GpM6JTbXTkF5HkjS1RWHUToDXh5/IeHaFnvRAacNCiqtoonpVS305JNzO0bdmJw+78z/tcxZG5fGHV7lSnfn8+NnwzEF/Qx7uL3E25ZtqoY2xKUiDiB0cD5wInAQBE5sfg1xpj/GGM6GGM6AA8A3xljsopdcmbh5yMrRaxs4/EFK2Uop6KKToDdneOt0L4sKxdGvLPkDTbsW8fInk+WejheOFJch+8Bs6IXBXB/939xRHpD7vvmNgLB+HmjURZjDPd/cztLdi5m9PnjOL5O81iHpGxmZw+qK7DWGLPeGOMFJgGXlHH9QGCijfEoC8Vq7ilcxc+iCidR5XkDluy12u/Zx3M/PUmvo3vT+9hzI25HgPTkwytOuF0OS062rZlci0dPf5YlOxcz7tfXo26vMoz79XU+WPEu/zjln5x73AWxDkdVAjsTVGNgS7GPtxY+dhgRSQP6AFOKPWyAmSKyUERuti1KVWEenzUv5pUhECz/0ESrDiIE+O8vz7LPs5eRPZ+Mam4kxV16BY0ayS6smHW5pEV/zjjmbJ6e+wjbc7ZZ0KJ9fv79R0Z+dy/nNLuAf5zyYKzDUZXEzgRV0u9Qaa9qfYEfDxne62GM6URoiHCYiPQq8SYiN4vIAhFZsGvXrugiVmGx4sjzylbWoYlWHUS4ef8mxmaO5soTr6XNEe0jbkeAdHfpQ3kOh5BuwVCfiPBU7xfwBbz867v7o27PLttztjHk02s5uuaxvNznzSpXEVyVzs7/6a1A02IfNwFKe5s2gEOG94wx2wr/3glMJTRkeBhjzBhjTGdjTOcGDRpEHbQqWyL1nkriL0xUWbleCvwBvP5g2OdXleepH0fiFCf3dx8ZVTupbme51TLSwrgmHM1qH88dXe9n2m+T+XbjV1G3ZzVvwMuQT68hx5fNm30nUSuldqxDUpXIzgQ1H2guIs1ExE0oCU0/9CIRqQWcDkwr9li6iNQo+jdwLrDMxlhVmKwaCos1XyDIvjwf+yw4JRdg0fb5TF39AUNPvoOjajSJuJ3yek8Hr7No2TnAsM53c3yd5jww607y/fmWtGmVh2ffy/w/fuL5c16jdf2TYh2OqmS2JShjjB8YDnwJrAQ+MMYsF5GhIjK02KWXATONMbnFHmsI/CAivwK/AJ8ZY2bYFasKT4E/EPWpsvHGiq/GGMNjcx6gftoRUW8aTUsOvzJGSpITtwXLzpNdyTzd+79s3L+eF3/5T9TtWWXi8nd4e8kYbj35Li5p2T/W4agY0GKxKmxZud5K2YiZaL5YO53Bn1zFs2e9xKB2N0Xcjgg0yEiu0OIKf+GRIFb8Fg/7YjCfrPmIWdfO54S6LSxoMXKZ2xdyyQdn0fWo7ky8fHpUy/WVffS4DRUXvP6gJqcS+AI+/v39P2let1XUx7inu10VXvnncjpIdVtzAOIjvZ4mxZXKiFl3xLRI6a68ndz46QAapDfktQvf0eRUjWmCUmGpKnNPVntn6VjW71sb9aZchwhpESaajGQXVlT7aZDekAd7PMYPW2bz0apJ0TcYAX/Qz9DPB7EnbzdvXTSJeqn1YxKHig+aoFS5fIEgXu09HeZAwX6em/cEpzU9g7Ob9YmqrVCSiSzLiAg1kpOiun+R69reSMdGnfnXnBHs8+y1pM2KePyHh/hxy3c8c9ZLtGvYsdLvr+KLJihVLu09lezFX/7DXk8W/+r1VFSbcp0OiXqYLtXtxGXBsvOiYrJZ+bt56sdHom6vIj5e/QGvLfwvg9v/natOurZS763ikyYoVSZ/IGhJAdWqZsuBTbyx+GX6t76atkd0iKotq5aL10ixphfV9ogO3NThVt5Z8gaL/vjFkjbLs2LXUu6aeQtdjzqVR09/1rb7iEC9dDe1UpNIdunLX7zT/yFVpniqWB5Pnv7xEQRhRI9/RdWOyyGW1NaDwjp9Lmvauq/7SBqmN+K+b27HH7S3B73Ps5fBn1xFzeSajL1oAm4bjzmvmZKEyxmqZ1g7zU2DjGQykl2WbHpW1tMEpUoVCBrLqixUJb/uWMSUVZP4+8m307hG0/KfUAYrShYVl5FiTZ2+DHcN/n3G/7Fs16+My3zNghZLFjRBbv1iMNuyt/LmRRM5Ir2RbfdKdTsPezNQVDaqfkYyddLcpCQ5Lfn+KWtoglKliveK5bFQdFJuvdQGDO/8j6jaSnJaU5m8OKdFdfoALmp+Gb2PPY+n5z7KHzm/W9Lmof4z73FmbfySx894js5HnWLLPSDUU61RzvfF7XJQKzWJBjWSqZmSZMnZWyo6+j+gShQIGjwVOE/pUFVpA3hxX63/nLlb53DvqQ9RI7lmVG1ZNfd0KKvq9IkIT535PIGgn4dn32tBZH81Y90nPP/zUww4aVBUG5zLIwK109xhL2QRCS1aqZvupl66mzS3E4ee2hsTmqBUifK8kVf43pHzBye9fjRXT72UFbuWWhpXLPkCPh77/kGa123JNW0GR9WW2xk6yt0OVtbpO6Z2M+7sNoJP10zlmw1fWtImwNqs3xg+40baN+zE073/a+ux7TVTkiJO2C6ngxopoV5V7bTQwgpNVZVHE5Q6TDBoKnQa7aHGZr7C3vw9LPzjZ856txu3fzmErQc2WxhhbLy3bBxr9/7Gw6c9QZIzuhVzGSn2Vkewqk4fwC0n30nzui0tKyab481m8CdX4XYm8+ZFE0lxRV8qpzQlzTtFKtkVWlhRPyOZGikuS5b1q7JpglKHyfMFIu49ZRcc4O0lb3Bh80v56Ybl3HLynUxb/SE9xrfjkTkj2OvJKr+ROJRdcID/zHuc7k16cU6Up7kmuxyVMr9Rw6IkGCom+yKbD2zkvz8/E1Vbxhju+HII6/euYcyF79Kk5tGWxFiScOadIuFwCGluF/Uykqmb7ibV7bSkkoc6XLm/JSJykYieEFZdGGPIi2JxxLvL3uJAwX6Gdb6bOil1GdnrSeYOXsqlLa/k9YUv0u2tE3lp/v/F3bEO5Xl5wXPsyd8V9aZcsG/u6VBW1unr0bQXV7S+htELRvHbnlURt/Py/P/js7XTeLjnE5zW9HRLYitJReedIpXkdFAzJYkGGcnUSk2yrNeqQsL5bg4A1ojIsyLS2u6AVGzleQNEur7BG/AyZtFL9Gh6Oh0b/VmYuHGNpvz3vDHMuu4Xuh7VnSd+eJju49owYdnbBILxv4z99+wtvL7wRfq3Gkj7hp2iaislqfSj3O2Q4bamTh/AyF5Pkp6UEXEx2dmbvuapuY9wacsr+Hun260JqhTRzDtFQiS0n61OemgIMD3ZpQsrLFDub4ox5lqgI7AOGCci8wqPWa9he3SqUhljolpa/tGq9/kjZ1upZyK1rt+Gdy/9iI+umEmjjKO4+6uh9H63KzPXfRbXq/6e/vFRDIYRPR6Jqh2h8npPRRwO6+r0NUg7gn+e9m/mbp3D5JUTy39CMZv2b2ToZ4NoWe9EnjvnVVt7NlbOO0XC6QgtUilaWJHi0r1VkQrrrZwx5gAwBZgEHEnokMFFInKbjbGpSpbvi7z3FDRBXlnwPCfWb8uZx5xT5rXdm/Tk8wFzGHvRBHwBL4Om9+eyD89hwbafIru5jZbsWMzklRO4udNtUc+XpFi0/LuirKrTB3BN28GcfGTXCs0n5vnyuOGTqzAYxvV9n/SkdEtiKYld806RSnY5qZWWpAsrIhTOHFRfEZkKzAKSgK7GmPOB9sA9NsenKokxJqqyRl+v/4LfslYyrPNdYb07FhEuan4Z3w1axDO9X2Td3rVc9P6Z3PjJANZm/RZxHFYyxvDo9w9QJ7Uet3WJ7kddCA23xYpVdfoc4uDZs15inyeLJ38YWe71xhju/XoYK3Yt5ZXzx3Ns7eMsiaMklTXvFAldWBGZcHpQVwDPG2PaGWP+Y4zZCWCMyQNusDU6VWk8viDBKIbZRi8YReMaTbm4RcWO5k5yJnF9+yH8NHgZ9506ktmbvuH0dzpx39e3sSPnj4jjscLXG2bw45bvuOeUf1IzuVZUbaW6nWEf5W4HK+v0ndSgHTd1HMb/lr5Zbq/3zcxXmLJqEvd1H8lZzc6z5P6lqex5p0jpworwlXvku4g0A/4wxngKP04FGhpjNtofXsXoke+R251TQCAYWYKav20efd/vzeNn/B83dRwWVRy78nbyws9P886SsbgcSQw9+XZuPfmuqKs2VJQ/6OfM/3UhaALMvm5hVPueRKB+enJMExSEqoPsySmw5Hj4HG82vd7uSO2Uusy8Zm6JhzXO2/oD/Sf34Zzjzuetvu/jsHExcKrbSU2LeomxEAga8n0BPL5AxL+HsRAPR75/CBQ/byFQ+JiqIqL9pRi9YBR1UuoyMMojzyE0Ef/EmaP4/vpMzjv+Qp7/+Wm6jTuJNxa9TIG/IOr2wzVh2XjWZK3i4Z7Rb8pNd7tinpwgNHmfZtH8TIa7Bo+f+Rwrdi9l7OLRh31+W/ZWhnx2DcfWPo4Xzxtra3JKcjriat4pEkULK+pn/LmwwiGCCHH9x+6ViuH0oDKNMR0OeexXY0x7OwOLhPagIhNN72lN1mp6vt2Bu7o9wP3dy5+TqKjM7Qt5/IeH+GHLbI6ueSwjevyLS1teaesLXo43m1PGteGEOi2YesXMqOY0RKBBRnLczIsYY9id441qOLd4W4Om9efHrd/x/fWLD1Z2L/AXcNmH57B6z0q+GPg9Leq1ivpepQmd75ScEEN7qnTR9KB2icjFxRq6BNhtZXAqdqLtPb268AVSnCnc2OEWC6P6U4dGJ/Nhv8+ZeNl0aiTX5NYvBnPehB58t+kbW+4HoR7h7rydlm3KjZfkBIXHw1tUYUJEeOLMURgT5OHZfy4i+ee3d7No+3xePO8NW5MTJM68k4pMOAlqKPCgiGwWkS3A/cDf7Q1LVZa8KGrubc/ZxuSVExjQ5nrqpzWwMKq/EhHOPPYcvrpmHqP7vMU+z16u+ugirppyEUt2LLb0Xtuyt/Lawv9yeaur/rLZOBIOEVJjuB+nNFbW6Tu61jHc3e0BPl87nZnrP+fdpW/x7rK3uL3LvVzY/FJL7lGaWO93UvYrd4jv4IUiGYXXZ9sbUuR0iK9iCvwB9uX5In7+v7//J68ufIF5f1vGMbWbWRhZ2Qr8Bby9ZAwv/PwMWZ49XN7qKu7v/gjH1Do26rbv+PJmpq5+nx+uX8LRtY6Jqq2aKUmWlRqymj8QZE+u15K2vAEvZ797CvsL9rLXk0X3Jr1479KPcTrs+9qTnA7qpCXFVe9URS6aIT5E5ELgVuAuERkpItZPNqhKlxfFvqcDBft5Z8lY+ja/vFKTE4SKl97c6TZ+umE5d3S9jy/WTue08e14ePY97M7bFXG7y3b+ygcr3uWmDsOiTk5Oh8RtcgJr6/S5nW6eOetFduRup2H6kbxy/nhbk5MI1ErV5FQdhLNR9zXgKuA2QvsNrwCi++1VMef1B/EGguVfWIp3lowl23uAWzvfZWFUFVMzuRYP9HiUuYOXcuWJ1/Jm5qucMu4kXvj5GXJ9uRVqq+ik3Nopdbij631Rx1bZJY0iYWWdvlObnMY7l0xhcv8Z1E2tZ02jpdB5p+ojnB5Ud2PMIGCvMeZR4FSgqb1hKbtFU7G8wF/AG4tfpmfTM6MunmqFIzMa89w5rzD7uoX0bHoGT899hFPfasM7S8biD4b3dc7aOJPvt3zLP055kFoptaOKx+WQhJgbcTisO9gQ4NzjLrBkmLUsaTrvVK2Ek6A8hX/nichRgA+o3DEdZSlfIEiBP/Le05RVk9iRu53hXf5hYVTRa1GvFeMu/oDpV37DMbWO5b5vbuOMd07m87XTyixG6w/6eXTOAzSrfTyD2g2JOg67DyO0Upo7cerDJTkdCdEzVdYJJ0F9IiK1gf8Ai4CNQMVKGau4Es3cU9AEeXXh87Rp0J5eR/e2MCrrdG3cnelXzWL8xR8gItzwyQD6vn8mP//+Y4nXT1r+Dr9lreSh0x7H7XRHde8kp4Nki0oKVRar6vTZSeedqqcyE1ThQYXfGGP2GWOmEJp7amWM0UUSCcofCOLxR56gZq7/jDVZq8MuChsrIkKf4/vy7XULeO7sV9h6YDOXfHA2g6b1Z9XuFQevy/Xm8Mzcx+h61KlccMIlUd83Ed/hW1mnzy4671Q9lZmgjDFB4LliHxcYY/bbHpWyTW4U+54ARs8fRdOax9C3RT+LIrKXy+HimraDmTt4Gf/s8Rjztn5P73e7cNfMoWzL3sroBc+zK28Hj/R6OuqEm+xy4HYlZuHPjBRX3J5ZpPNO1Vc4v00zRaSfRPDbKyJ9RGS1iKwVkRElfP5eEcks/LNMRAIiUjec56qKCwQNHl/kCeqX3+cy/4+fGHryHSUWB41naUlp3Nb1Xn6+YQVDOg5jyqqJdB/XllcWjOKSFv3pdGTXqO+RnoC9pyJW1umzks47VW/h1OLLBtIBP6EFEwIYY0yZ5aVFxAn8BpwDbAXmAwONMStKub4vcJcxpndFn1tEN+qW7YDHR34UPahB0/qzYNtPzL9pta2HzlWGzfs38ey8x/hh87dMu/KbqPdypRQeTJfIrKzTZwWts1d9lLZRt9y3JsaYSI927wqsNcasLwxgEnAJUFqSGcifiy8q+lxVjmDQ4IkiOa3es5KZ6z/jnlMeSvjkBKESPS/3edOy9tKTE38IqqhO3/78yKuLWEnnnVS5CUpEepX0uDFmTjlPbQxsKfbxVqBbKfdIA/oAwyN47s3AzQBHHx3dkdxVWa7XH9U5QK8ufIFUVyqDO2gZxkOlJDlxVZFD51KSnOR5A/ii2MRtBZ13UhBGggLuLfbvFEK9m4VAeWuMS3rrU9prZF/gR2NMVkWfa4wZA4yB0BBfOTFVS8GgiWpob1v2VqasnMh17W6kXmp9CyNLfEJirtwrS40UF1kW1emLRJLTkRBL35X9whni61v8YxFpCjwbRttb+WvFiSbAtlKuHcBf91ZV5LmqHHm+QFS9pzcWjyZoggztdKdVIVUZqW5nlRuGSiqs0xfNm5pIFe13UgrCLBZ7iK1AmzCumw80F5FmIuImlISmH3qRiNQCTgemVfS5qnzGmKjKGu337ON/S9/k4hb9oi6gWtUIodNyqyIr6/RVhM47qeLCmYN6iT+H1xxAB+DX8p5njPGLyHDgS8AJvGWMWS4iQws//1rhpZcBM40xueU9N+yvSh2U5w0QzaKsd5a8QY43O6ZFYeNVWnJ8HOVuh6I6fdmeyN/cVJTOO6lDhbPM/PpiH/qBjcaYkmvGxJguM/8rYwy7cgoiTlAev4eub7bmxAZtmHT5J9YGl+Di7Sh3u+zOKYjqxOVwJTkd1E2PrsyUSlwRLzMHJgMeY0ygsCGniKQZY/KsDlJZK98XXe9pysqJ7MzbzujOb1kXVBWR7o6vo9ztUiPFFdWhluHQeSdVmnDmoL4BUot9nAp8bU84yirGGHKjKAobCAZ4ZeHztDuiI6c1PcO6wKoAhwhpcXwYoZWSXU6SbS7fpPNOqjTh/OSlGGNyij4o/HeafSEpK3h8wagqAny5/lPW7V3DsM53V4ueQkWkJzur1fekRkqSbXX6dN5JlSWcBJUrIgdPpRORk4F8+0JSVsiNYuWeMYaX5z/HMbWacWHzSyNqo6q+fjsdQmo1e0G1q06f7ndS5Qnnp+5O4EMRKdqHdCShI+BVnPL4AlFNbP/0+48s2j6fp3q/EHFR2KJhm2yPP+ZVCayUkVw95p4OlV64L8qqOn0676TCEc5G3fki0gpoSWjrxypjTHwU61Ilyi2Ibmnw6AWjqJfagAEnDYro+Q7588jzuuluPL4AOQX+SlkNZidnghzlbger6/TpvJMKR7lDfCIyDEg3xiwzxiwFMkTkVvtDU5Hw+AL4o0gEK3cv5+sNX3Bjh1tIdaWW/4QSHFo4NSXJSb10d2HvI+LQYq6qlTSqqJQkJ0kW1BzUeScVrnB+2oYYY/YVfWCM2QsMsS0iFZW8KMvThIrCpvG39jdH9HwRSpyjERHSk13UT08mNQFXwCU5HfqiSmjZeTR03klVRDgJylH8sMLCs5qq5I66/Xk+9uf78PoTc86kwB9dFerfs7fw0apJXNt2MHVT60XURmpS2SvcHA6hZkoS9dLduBOoAnhVOE7DCtEkap13UhUVztuhL4EPROQ1QiWPhgIzbI0qRoLG4PUH8fgChfMoDsuGNSpDXhT7ngDGLHoZYww3d7otoucLkBZmbTqX00GddDcF/gDZnvien3I7HSS7NEEVqZHsosBf8U3gOu+kKiqcV5P7CZ23dAuh16CZwBt2BhUPgsaQ5w2Q5w0cnBxPcTni9twfrz+IN4re0z7PXt5d+haXtryCpjUjKwqb7Kp4Ze9kl5PkDCd5Xj85Bf6oKl/YJZGPcrdDJHX6dN5JRaLcV1tjTNAY85oxpr8xph+wHHjJ/tDiRyBoyC3wsyfXS1aulzyvn2CcveOPpmI5wNtL3iDXlxNVUdi0KIbB0tyh+ak0t9O2TaGRSHY5cNtcSSERpbldYb8Z0XknFamw3hqKSAdCR7JfBWwAPrIxprjmCwTxBYJk48ddOB6fkuSI6d4YXyBIQRTzZh6/hzcWj+bMY8/lpAbtImojyemIeijU4RBqpCSRmuQkp8Af1ddkleq+cq8s4dTp03knFY1Sf/tEpAWhc5gGAnuA9wlVPz+zkmKLe95AaFgt2wNuVyhZJbsqP1lFO/f04Yr32J23k2Gd7464DStr07mcDmqnufH6g2R7fFEtm49GiqvqHOVuh1CdvkCZbyR03klFo6y3h6uA74G+xpi1ACKihwKVwAAF/lAvRiT0i5uSVDkT6/5AEI8/+qKw7Rt2okeTXhG1YdcGVrfLQb2MZPK9oY2+VlUxCIcAGVEuqa4OMpJdeP3eEk9s1nknFa2y3h72A7YD34rIGyJyFsTV9EBcMia0WXZfno+d2R4OeOxdtp4b5b6nL9ZNZ8O+dQzv/I+Ie352V/ZOdTupn+EmPdlVaT+AKVXwKHc7uAqPhz+UzjspK5SaoIwxU40xVwGtgNnAXUBDEXlVRM6tpPgSmjGQ7w2wN8/L7pwCcgr8+C2sSxcIGgp8kSeooqKwx9Y6jgtOuCSiNkrbmGs1kdDKsXoZyaTY3DOtyke52yEj2YWj2JsbnXdSVglnFV+uMeY9Y8xFQBMgExhhd2BVTfGVgHtyCsi1oDZdrtdf4tBKuOZu/Z7MHQu55eQ7cToie9Evb2Ou1ZwOoVZaEnXS3LbtT0vV3lOFFNXpK1IrVeedlDUq9BtujMkyxrxujOltV0DVgT9oyCnwszunIOJl68GgwRPl8N7oBaOon3YEV550bUTPr8jGXKu5XaEjwmulJv3l3Xu0RLT3FImiDe1pbqdualaW0d/EGCtatp6Dv0IrAaPtPa3YtZRZG79kRPdHIi4KG8nGXKsVfb/yvAFyC6L7nkAo4Tr03X9Eaqa4dNWjspQmqDjxl5WAFC7hTSq57lkwaMiPsvf0yoLnSUtK5/r2kdf9jWZjrpWKCtGmJjnJLvDjiXBeLtR7io+vKRFpclJW05+oOGQAjz/A/vySVwLm+QJR9RS2HNjE1NUfcF3bG6iTUjeiNtwWbMy1msMh1EpNom6EhWir62GESsUr7UHFuaKVgPnePwvY5kexcg/gjUUvIyIM6Tg84jbi+ciMpMJCtBU5KNEh1e8od6XinSaoBFJUwDYaez1ZvLtsHJe2vJImNY+OqI1EOVm2aH4qvzBRlbXPV3tPSsUfTVDVzPhfx5Dny2VYNEVh47j3dCgRIc3tIsXlJMfrL3HuzumQuO4RKlVdxdckgrJVvj+fsYtf4axmfWhdv01EbVTWxlyrFT8oMfmQ6uRaEFap+KS/mdXIB8vfZU/+riiLwib2UFhRIdqigxIFEmK4UqnqSBNUNREIBnh14Qt0atSFUxufFlEbAqRVkRfzooMS4/kkX6WqOx3iqyY+W/sxG/evZ1jnuyPuASUnOavcJtZYbzRWSpVOE1Q1ECoKO4rjap9An+P7RtyObmJVSlUmWxOUiPQRkdUislZESiwwKyJniEimiCwXke+KPb5RRJYWfm6BnXFWdT9u+Y4lOxdxa+e7Ii4K63Y6tFKAUqpS2TYHJSJOYDRwDrAVmC8i040xK4pdUxt4BehjjNksIkcc0syZxpjddsVYXYxeMIoGaQ3p3/rqiNuIl7JGSqnqw863xF2BtcaY9cYYLzAJOPTQoauBj4wxmwGMMTttjKdaWr5rCd9u+oohHYeR4kqJqA2nQ7RCtVKq0tmZoBoDW4p9vLXwseJaAHVEZLaILBSRQcU+Z4CZhY/fXNpNRORmEVkgIgt27dplWfBVxegFo0hPyuD6dpEXhdXjJ5RSsWDnK09Jy6MOXdPrAk4GzgJSgXki8pMx5jeghzFmW+Gw31cissoYM+ewBo0ZA4wB6Ny5s64ZLmbz/k1MWz2ZIZ2GUyuldkRtiEBKks49KaUqn52vPFuBpsU+bgJsK+GaGYWn9u4G5gDtAYwx2wr/3glMJTRkqCpgzKIXcYiDm6MoCpvoG3OVUonLzgQ1H2guIs1ExA0MAKYfcs00oKeIuEQkDegGrBSRdBGpASAi6cC5wDIbY61ysvL38N6y8VzeagBH1WgSURtVaWOuUirx2DbEZ4zxi8hw4EvACbxljFkuIkMLP/+aMWaliMwAlgBBYKwxZpmIHAdMLXzn7gImGGNm2BVrVTTu19fJ9+dxy8l3RtxGVdyYq5RKHGLKOoMgwXTu3NksWBD5lqm9uV68gWD5F8a5PF8end9sycmNuvK/S6dE3E69dLfufVJK2U5EFhpjOh/6uL76VEGTlr9DVv5uhneJvCisbsxVSsWavgJVMf6gn9cW/pfOR3aj61HdI25HN+YqpWJNE1QV8+maqWw+sDGqorAu3ZirlIoDmqCqEGMMryx4nhPqtOC84y+KuJ003ZirlIoDmqAKBU3iL474fvO3LNm5mFs734VDIvuv1Y25Sql4oa9EhcYuGkuvt7vw6sIX2J5z6H7ixDB6wSgapjeiX6uBEbehG3OVUvFCE1ShBmkNcDuTeXTOA3R84wSunHIhH6x4jxxvdqxDC8vSnZl8t/kbhnQcTrIrOaI2dGOuUiqe6D6oYvbmelm5ezVTVk5iyqqJbNq/gVRXKn2O70v/1ldz+jFn4XLE5/zM0M8H8fWGGSy6aQ01k2tF1Eaq20nNlCSLI1NKqbKVtg9KE1QxxTfqGmNY8MdPTF45iem/TWavJ4v6aUdwaYv+9Gs9kA4NT46bobBN+zdy6riTGNrpDkb2ejLidnRjrlIqFjRBhaG0ShLegJdZG79k8sqJfLX+cwoCBZxQpwX9Wg/k8lYDOKbWsVFEHb0Hv72L/y15k/k3rqJRxlERtZHsclA7zW1xZEopVT5NUGEIp9TRfs8+Pl0zlcmrJjJv6/cAdDuqO/1aD6Rvi8upk1I34vtHYnfeLrq82ZJLW17J8+e+FnE7tdOSdO+TUiomtNSRRWql1OaatoOZesVM5t+4igd7PEaWJ4v7vrmN9mOaccMnV/HZmo8p8BdUSjyhorD5URWF1Y25Sql4pD2oYiItFmuMYdmuX5m8cgIfrfqAXXk7qJVcm77NL6d/64F0bdw94n1JZcn15dJlbEu6Nj6V8Rd/GHE7tVKTSNHVe0qpGCmtBxWfS9ISjIjQ9ogOtD2iAw/3fJIftsxm8ooJTFk1iXeXvUWTmkfTr9UA+re+muZ1W1p234nL3ibLs4dhnSMvCusQIdmlHWmlVPzRHlQxVh+3kevNYca6T5i8ciLfbf6GoAnS7ohO9G89gMtaXkmD9IYRt+0P+jll3EkcldGY6VfNiridjGQX6cn6PkUpFTs6BxUD6e4M+rUeyMTLp5M5ZB2Pnf4sACO/u48ObxzPwKmXMGXlRHJ9uRVu+5PfprD1wGaGd/5HxPEJkKpDe0qpOKU9qGIq68DC1XtW8tGqSUxeOZHfs7eQlpTOBSdcQv/WA+nZ9EycjrKThjGGs987BW/Ay3eDFkY8v6Ubc5VS8UCXmYehsk/UDZogP/8+lykrJzJ9zRQOFOzniLRGXNbqSvq3HkibBu1L3Aw8e9PXDPioL8+f+zoDTxoU8f11Y65SKh5oggpDLI989/g9fL3hC6asnMjXG2bgC/poUbc1V7QeyGWtrqJJzaMPXnvF5AtYk7WKn29YGXHdPd2Yq5SKF5qgwhDLBPWXODxZfPLbR0xeOYFfts0D4NQmPenfaiDH1GpG/ynnM7Lnk9za+a6I71EnzY1bV+8ppeKAJqgwxEuCKm7Tvg18tPp9Jq+cwLq9awCo4a7JopvWUCO5ZkRtuhxCvYzIel5KKWU13QeVoI6p3Yy7uo3gzq73k7ljIR+v/oD2DTtFnJwAXVaulEoI+kqVIESEjo0607HRYW8yKsQholUjlFIJQSchqpk0tyYnpVRi0ARVjejGXKVUItEEVY2kuJ04HPFxyKJSSpVHE1Q1ku7WKUelVOLQBFVNJLscOLX3pJRKIJqgqok07T0ppRKMJqhqIMnp0KoRSqmEY+urloj0EZHVIrJWREaUcs0ZIpIpIstF5LuKPFeFR5eWK6USkW3jPiLiBEYD5wBbgfkiMt0Ys6LYNbWBV4A+xpjNInJEuM9V4dGNuUqpRGVnD6orsNYYs94Y4wUmAZcccs3VwEfGmM0AxpidFXiuCoP2npRSicrOBNUY2FLs462FjxXXAqgjIrNFZKGIDKrAcwEQkZtFZIGILNi1a5dFoVcNgiYopVTisnNpV0lrmg8tne4CTgbOAlKBeSLyU5jPDT1ozBhgDISqmUccbRWU6naWeOChUkolAjsT1FagabGPmwDbSrhmtzEmF8gVkTlA+zCfq8qhS8uVUonMziG++UBzEWkmIm5gADD9kGumAT1FxCUiaUA3YGWYz1VlSHE5dWOuUiqh2fYW2xjjF5HhwJeAE3jLGLNcRIYWfv41Y8xKEZkBLAGCwFhjzDKAkp5rV6xVUarOPSmlEpyeqFtMPJ6oG4kkp4O66e5Yh6GUUmEp7URdLS9QBenKPaVUVaAJqorRjblKqapCE1QVk56syUkpVTVogiom0bcMieiJuUqpqkM3yhRTO81NgT+AxxukwB8oeWdwHEtN0o25SqmqQxPUIZJdTpJdToJBFwX+IPm+AL4EWdmnG3OVUlWJvqKVwuEQUt1OUt1O/IEgHn+QfG+AYJwuy9eNuUqpqkYTVBhcTgcZTgcZyS68hb2qAl98DQGm6eIIpVQVowmqgtyu0Om0JqVwCNAbiPnm3iSngySnrndRSlUtmqAiJIX7jVKSnASCBo8vQL4vQCBY+f0q3ZirlKqKNEFZwOkQ0pNdpCe78AVCQ4AeX4DKmK5yOnRjrlKqatIEZbGi4bYayaEhQI8vgNcftG2+SntPSqmqShOUTYoPAQaDBo8/QL43gN/CIUDdmKuUqso0QVUCh0NIc7tIc7vwHxwCDEa9ZF035iqlqjJNUJXM5XRQw+mgRgpRVa0QdGOuUqpq01e4GCqqWmGMC4+vYlUrknVjrlKqitMEFQdE/qxaEQiag6sAy1qyrhtzlVJVnSaoOON0CBnJrr9WrfD/dcm6WzfmKqWqAU1Qcexg1Qrz55L1An+QVF1arpSqBjRBJYBDl6w7dO5JKVUN6DhRgtHkpJSqLjRBKaWUikuaoJRSSsUlTVBKKaXikiYopZRScUkTlFJKqbikCUoppVRc0gSllFIqLmmCUkopFZc0QSmllIpLYqI8NC+eiMguYFMUTdQHdlsUjp00TuskQoygcVotEeJMhBjBmjiPMcY0OPTBKpWgoiUiC4wxnWMdR3k0TuskQoygcVotEeJMhBjB3jh1iE8ppVRc0gSllFIqLmmC+qsxsQ4gTBqndRIhRtA4rZYIcSZCjGBjnDoHpZRSKi5pD0oppVRc0gSllFIqLmmCAkTkLRHZKSLLYh1LWUSkqYh8KyIrRWS5iNwR65gOJSIpIvKLiPxaGOOjsY6pLCLiFJHFIvJprGMpjYhsFJGlIpIpIgtiHU9JRKS2iEwWkVWFP5+nxjqmQ4lIy8LvYdGfAyJyZ6zjKomI3FX4+7NMRCaKSEqsYzqUiNxRGN9yu76POgcFiEgvIAd4xxjTJtbxlEZEjgSONMYsEpEawELgUmPMihiHdpCICJBujMkRkSTgB+AOY8xPMQ6tRCJyN9AZqGmMuSjW8ZRERDYCnY0xcbtpU0TeBr43xowVETeQZozZF+OwSiUiTuB3oJsxJprN/ZYTkcaEfm9ONMbki8gHwOfGmPGxjexPItIGmAR0BbzADOAWY8waK++jPSjAGDMHyIp1HOUxxvxhjFlU+O9sYCXQOLZR/ZUJySn8MKnwT1y+CxKRJsCFwNhYx5LIRKQm0At4E8AY443n5FToLGBdvCWnYlxAqoi4gDRgW4zjOVRr4CdjTJ4xxg98B1xm9U00QSUoETkW6Aj8HONQDlM4bJYJ7AS+MsbEXYyFXgDuA4IxjqM8BpgpIgtF5OZYB1OC44BdwLjC4dKxIpIe66DKMQCYGOsgSmKM+R34P2Az8Aew3xgzM7ZRHWYZ0EtE6olIGnAB0NTqm2iCSkAikgFMAe40xhyIdTyHMsYEjDEdgCZA18LhgLgiIhcBO40xC2MdSxh6GGM6AecDwwqHpOOJC+gEvGqM6QjkAiNiG1LpCocgLwY+jHUsJRGROsAlQDPgKCBdRK6NbVR/ZYxZCTwDfEVoeO9XwG/1fTRBJZjCeZ0pwHvGmI9iHU9ZCod5ZgN9YhtJiXoAFxfO70wCeovIu7ENqWTGmG2Ff+8EphIa948nW4GtxXrKkwklrHh1PrDIGLMj1oGU4mxggzFmlzHGB3wEdI9xTIcxxrxpjOlkjOlFaIrE0vkn0ASVUAoXILwJrDTGjIp1PCURkQYiUrvw36mEftlWxTSoEhhjHjDGNDHGHEtouGeWMSau3qUCiEh64YIYCofNziU0vBI3jDHbgS0i0rLwobOAuFm4U4KBxOnwXqHNwCkiklb4O38WofnmuCIiRxT+fTRwOTZ8T11WN5iIRGQicAZQX0S2Av8yxrwZ26hK1AO4DlhaOMcD8KAx5vPYhXSYI4G3C1dJOYAPjDFxu4Q7ATQEpoZep3ABE4wxM2IbUoluA94rHD5bDwyOcTwlKpwvOQf4e6xjKY0x5mcRmQwsIjRstpj4LHs0RUTqAT5gmDFmr9U30GXmSiml4pIO8SmllIpLmqCUUkrFJU1QSiml4pImKKWUUnFJE5RSSqm4pAlKqTgmIjnF/n2BiKwp3HeiVJWn+6CUSgAichbwEnCuMWZzrONRqjJoglIqzolIT+AN4AJjzLpYx6NUZdGNukrFMRHxAdnAGcaYJbGOR6nKpHNQSsU3HzAXuDHWgShV2TRBKRXfgsCVQBcReTDWwShVmXQOSqk4Z4zJKzy/6nsR2RGnhYyVspwmKKUSgDEmS0T6AHNEZLcxZlqsY1LKbrpIQimlVFzSOSillFJxSROUUkqpuKQJSimlVFzSBKWUUiouaYJSSikVlzRBKaWUikuaoJRSSsWl/weE12yjXa9BxwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best accuracy was with 79.81% with k = 5\n"
     ]
    }
   ],
   "source": [
    "#Calculting Accuracy for Various Ks\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn import metrics\n",
    "Ks = 10\n",
    "mean_acc = np.zeros((Ks-1))\n",
    "std_acc = np.zeros((Ks-1))\n",
    "\n",
    "for n in range(1,Ks):\n",
    "    #Train Model and Predict  \n",
    "    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)\n",
    "    yhat= knn.predict(X_test)\n",
    "    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)\n",
    "    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])\n",
    "\n",
    "# Plot model accuracy for Different number of Neighbors\n",
    "plt.plot(range(1,Ks),mean_acc,'g')\n",
    "plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)\n",
    "plt.legend(('Accuracy ', '+/- 3xstd'))\n",
    "plt.ylabel('Accuracy ')\n",
    "plt.xlabel('K')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print( \"The best accuracy was with\", '{:.2%}'.format(mean_acc.max()), \"with k =\", mean_acc.argmax()+1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Model Selection\n",
    "k = 5 \n",
    "knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Decision Tree\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# function for fitting trees of various depths on the training data using cross-validation\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):\n",
    "    cv_scores_list = []\n",
    "    cv_scores_std = []\n",
    "    cv_scores_mean = []\n",
    "    accuracy_scores = []\n",
    "    for depth in tree_depths:\n",
    "        tree_model = DecisionTreeClassifier(max_depth=depth)\n",
    "        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)\n",
    "        cv_scores_list.append(cv_scores)\n",
    "        cv_scores_mean.append(cv_scores.mean())\n",
    "        cv_scores_std.append(cv_scores.std())\n",
    "        accuracy_scores.append(tree_model.fit(X, y).score(X, y))\n",
    "    cv_scores_mean = np.array(cv_scores_mean)\n",
    "    cv_scores_std = np.array(cv_scores_std)\n",
    "    accuracy_scores = np.array(accuracy_scores)\n",
    "    return cv_scores_mean, cv_scores_std, accuracy_scores\n",
    "  \n",
    "#function for plotting cross-validation results\n",
    "def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):\n",
    "    fig, ax = plt.subplots(1,1, figsize=(15,5))\n",
    "    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)\n",
    "    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)\n",
    "    ylim = plt.ylim()\n",
    "    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)\n",
    "    ax.set_title(title, fontsize=16)\n",
    "    ax.set_xlabel('Tree depth', fontsize=14)\n",
    "    ax.set_ylabel('Accuracy', fontsize=14)\n",
    "    ax.set_ylim(ylim)\n",
    "    ax.set_xticks(depths)\n",
    "    ax.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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C8d/CGqXjRwEKqWK3gNbD/D27z3Gquv/zNu/9xgPbD0wocdrbGxk+fLlfPSjH2XlypV88IMf5OGHH+b0009nyZIlPPvsswD09fXxkY98hPXr13P88cfz+9//fmD7M844gxNOOIETTjiBp556CoCNGzeyYcMG/u7v/o7ly5fzwQ9+EGPMEfvfsWMHb3vb21izZg0nnHACO3fuZOPGjZx55plcdtllrFq1img0ypVXXsmqVas4/vjjeeSRRwB45ZVXOOmkk1i7di2rV69m+/bt9PX18c53vpM1a9awcuVK7rrrrkH7syyLhoYGurq6BpYtXryYlpYW7rvvPk4++WSOP/543va2t9HS0nJEe6+//npuvPFGALZs2cKaNWs49dRTuemmmwYd0+GOyXXXXcfjjz/O2rVr+c53vsPGjRt517veBUBHRwcXXnghq1ev5pRTTuGll14a2N9HPvIRNmzYwMKFC0fsPfz4xz/OunXrWLFiBV/60pcGlm/atInTTjuNNWvWcNJJJxEKhUilUnz2s59l1apVrF69mh/84AcANDQ00NbWBti9lRs2bBhow1VXXcW5557L5ZdfPuLzA/jGN77BqlWrWLNmDddddx07d+7khBNOGLh/+/btnHjiicM+h/HIZih6ErDDGLMLQETuBN4DvJqxjgGCYie4lgIdQDKLbZwWQtEETocQ8OT+SoRSSh0hc/6+oePyoj3Q/Pzh8X0HX4ZUwr6vdim86UK7t2/OOiity3rTlVL574GtB/i3323FMgafy8GBrij/9js7FfH8lTOP+XF37NjB//7v/3LLLbewfv16fvWrX/HEE09w77338p//+Z/cc8893HDDDZx11lncdtttdHV1cdJJJ/G2t72Nuro6HnroIXw+H9u3b+cDH/gAmzdvBuD555/nlVdeYdasWZx++uk8+eSTvPnNbx607w9+8INcd911XHTRRUSjUSzLYt++fTz77LNs3bqVBQsW8K1vfQuAl19+mddff51zzz2Xbdu2cfPNN/PpT3+aD37wg8TjcVKpFH/605+YNWsWf/zjHwHo7u4etD+Hw8F73vMe7r77bq688kr+9re/0dDQQH19PW9+85t55plnEBFuvfVWvvGNbwzsezhXXnklP/jBD3jrW9/K5z73uYHlIx2Tr3/969x444384Q9/AOxgud+XvvQljj/+eO655x7+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6uqa3AYpNQbahVOkRps8fqp0RxL0RBNUlXioC3pxaYVRpXKv5VV7OojtD9kTtR//IVj3DxCsz3XLlFKqaPQHgtdcc80R96VSqVEnFv/Tn/40lU07ZsYYjDE4HHo+V6z0lS1CkXiKrnAiJ/s2Btp747zREuJQKIplFe88lUrltf1b4Lcfg59fBHuegpP/Ea7aCGf+mwaBSikF9ljpX38Aelsn/FDXXXcdO3fuZO3atXzuc59j48aNnHnmmVx22WWsWrUKgAsvvJATTzyRFStWcMsttwxs29DQQFtbG42NjRx33HF87GMfY8WKFZx77rlEIpEj9nXfffdx8sknc/zxx/O2t72NlpYW++n09nLllVeyatUqVq9ezW9/+1sAHnjgAU444QTWrFnD2WefDcD111/PjTfeOPCYK1eupLGxcaAN11xzDSeccAL79u3j4x//OOvWrWPFihV86UtfGthm06ZNnHbaaaxZs4aTTjqJUCjEGWecwQsvvDCwzumnn85LL7004eOrpob2CBahA91Hfmlkm2VBS3eMjr449UEflSU6mbRSU84YO+j723/Dvk329A9v/mdYe5k9959SSqnDnr4Jdv4Fnv7hhCskf/3rX2fr1q0DQdDGjRt59tln2bp1KwsWLADgtttuo6qqikgkwvr163nve99LdXX1oMfZvn07v/71r/nJT37C+9//fn7729/yoQ99aNA6b37zm3nmmWcQEW699Va+8Y1v8K1vfYuvfvWrlJeX8/LLLwPQ2dlJa2srH/vYx3jsscdYsGABHR0dR30ub7zxBrfffvtAqusNN9xAVVUVqVSKs88+m5deeonly5dzySWXcNddd7F+/Xp6enrw+/189KMf5Y477uC73/0u27ZtIxaLsXr16gkdWzV1NBAsMj3RBH2x/Bmnl0gamjojtPfFmFHup9SrbzmlJp1lwc6/wt9uhoMvQ2md3fO3+v3g9ue6dUoplV1/vQFaXxv5/u0Pg5WwC2Y5nPDU9+1g0OGGJW8bfpva4+Csfx9XM0466aSBIBDg+9///sCk4Pv27WP79u1HBIILFixg7dq1AJx44ok0NjYe8bhNTU1ccsklHDhwgHg8PrCPhx9+mDvvvHNgvcrKSu677z7e8pa3DKxTVVV11HbPnz+fU045ZeD2b37zG2655RaSySQHDhzg1VdfRUSYOXMm69evB6Cs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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# fitting trees of depth 1 to 24\n",
    "sm_tree_depths = range(1,25)\n",
    "sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train, sm_tree_depths)\n",
    "\n",
    "# plotting accuracy\n",
    "plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, \n",
    "                               'Accuracy per decision tree depth on training data')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(criterion='entropy', max_depth=9)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Model Selection\n",
    "tree_model = DecisionTreeClassifier(criterion=\"entropy\", max_depth = 9)\n",
    "tree_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Support Vector Machine\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import svm\n",
    "#Fitting Models to determine the best kernal\n",
    "linear = svm.SVC(kernel='linear').fit(X_train, y_train)\n",
    "rbf = svm.SVC(kernel='rbf').fit(X_train, y_train)\n",
    "poly = svm.SVC(kernel='poly').fit(X_train, y_train)\n",
    "sig = svm.SVC(kernel='sigmoid').fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Linear Kernel: 76.92\n",
      "Accuracy Radial Basis Kernel: 76.92\n",
      "Accuracy Polynomial Kernel: 77.88\n",
      "Accuracy Sigmoid Kernel: 75.00\n"
     ]
    }
   ],
   "source": [
    "#Finding the most accurate kernel\n",
    "accuracy_lin = linear.score(X_test, y_test)\n",
    "accuracy_rbf = rbf.score(X_test, y_test)\n",
    "accuracy_poly = poly.score(X_test, y_test)\n",
    "accuracy_sig = sig.score(X_test, y_test)\n",
    "\n",
    "print('Accuracy Linear Kernel:', \"{:.2f}\".format(accuracy_lin*100))\n",
    "print('Accuracy Radial Basis Kernel:', \"{:.2f}\".format(accuracy_rbf*100))\n",
    "print('Accuracy Polynomial Kernel:', \"{:.2f}\".format(accuracy_poly*100))\n",
    "print('Accuracy Sigmoid Kernel:', \"{:.2f}\".format(accuracy_sig*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Fit Data onto Radial Basis Kernel\n",
    "svm_model = svm.SVC(kernel='rbf').fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "liblinear 0.711\n",
      "newton-cg 0.744\n",
      "lbfgs 0.744\n",
      "sag 0.744\n",
      "saga 0.744\n"
     ]
    }
   ],
   "source": [
    "#Finding best solver for our logisitic regression model\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']\n",
    "params = dict(solver=solver_list)\n",
    "log_reg = LogisticRegression(C=0.01, n_jobs=1, random_state=34)\n",
    "clf = GridSearchCV(log_reg, params, cv=5)\n",
    "clf.fit(X_train, y_train)\n",
    "scores = clf.cv_results_['mean_test_score']\n",
    "\n",
    "for score, solver in zip(scores, solver_list):\n",
    "    print(f'{solver} {score:.3f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creating our model\n",
    "LR_model = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Evaluation using Test set\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import jaccard_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import log_loss"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, download and load the test set:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2021-08-16 00:13:07--  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv\n",
      "Resolving s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)... 67.228.254.196\n",
      "Connecting to s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)|67.228.254.196|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 3642 (3.6K) [text/csv]\n",
      "Saving to: ‘loan_test.csv’\n",
      "\n",
      "loan_test.csv       100%[===================>]   3.56K  --.-KB/s    in 0s      \n",
      "\n",
      "2021-08-16 00:13:08 (84.5 MB/s) - ‘loan_test.csv’ saved [3642/3642]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Load Test set for evaluation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>50</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>9/15/2016</td>\n",
       "      <td>35</td>\n",
       "      <td>Master or Above</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>21</td>\n",
       "      <td>21</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/10/2016</td>\n",
       "      <td>10/9/2016</td>\n",
       "      <td>43</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/10/2016</td>\n",
       "      <td>10/9/2016</td>\n",
       "      <td>26</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>35</td>\n",
       "      <td>35</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>800</td>\n",
       "      <td>15</td>\n",
       "      <td>9/11/2016</td>\n",
       "      <td>9/25/2016</td>\n",
       "      <td>29</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           1             1     PAIDOFF       1000     30       9/8/2016   \n",
       "1           5             5     PAIDOFF        300      7       9/9/2016   \n",
       "2          21            21     PAIDOFF       1000     30      9/10/2016   \n",
       "3          24            24     PAIDOFF       1000     30      9/10/2016   \n",
       "4          35            35     PAIDOFF        800     15      9/11/2016   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0  10/7/2016   50              Bechalor  female  \n",
       "1  9/15/2016   35       Master or Above    male  \n",
       "2  10/9/2016   43  High School or Below  female  \n",
       "3  10/9/2016   26               college    male  \n",
       "4  9/25/2016   29              Bechalor    male  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df = pd.read_csv('loan_test.csv')\n",
    "test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>education</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "      <td>Bechalor</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>Master or Above</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>43</td>\n",
       "      <td>1</td>\n",
       "      <td>High School or Below</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>800</td>\n",
       "      <td>15</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>Bechalor</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender             education\n",
       "0       1000     30   50       1              Bechalor\n",
       "1        300      7   35       0       Master or Above\n",
       "2       1000     30   43       1  High School or Below\n",
       "3       1000     30   26       0               college\n",
       "4        800     15   29       0              Bechalor"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Preprocessing\n",
    "test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])\n",
    "test_df['due_date'] = pd.to_datetime(test_df['due_date'])\n",
    "test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek\n",
    "test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\n",
    "test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)\n",
    "test_df[['Principal','terms','age','Gender','education']].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>43</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>800</td>\n",
       "      <td>15</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   50       1        0         1                     0   \n",
       "1        300      7   35       0        1         0                     0   \n",
       "2       1000     30   43       1        1         0                     1   \n",
       "3       1000     30   26       0        1         0                     0   \n",
       "4        800     15   29       0        1         1                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "3        1  \n",
       "4        0  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Feature Selection\n",
    "feature_test = test_df[['Principal','terms','age','Gender','weekend']]\n",
    "feature_test = pd.concat([feature_test,pd.get_dummies(test_df['education'])], axis=1)\n",
    "feature_test.drop(['Master or Above'], axis = 1,inplace=True)\n",
    "feature_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Split X and y\n",
    "test_X = feature_test\n",
    "test_y = test_df['loan_status'].values\n",
    "\n",
    "\n",
    "#Scale X\n",
    "scaler = preprocessing.StandardScaler()\n",
    "test_X = scaler.fit(test_X).transform(test_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(54, 8)\n",
      "(54,)\n"
     ]
    }
   ],
   "source": [
    "print(test_X.shape)\n",
    "print(test_y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN Train set Accuracy:  79.34%\n",
      "KNN Test set Accuracy:  79.81%\n"
     ]
    }
   ],
   "source": [
    "#KNN Accuracy\n",
    "yhat_knn = knn.predict(test_X)\n",
    "knn_jaccard = jaccard_score(test_y, yhat_knn, pos_label='PAIDOFF')\n",
    "knn_f1 = f1_score(test_y, yhat_knn, average='weighted')\n",
    "\n",
    "knn_train_acc = metrics.accuracy_score(y_train, knn.predict(X_train))\n",
    "knn_test_acc = metrics.accuracy_score(y_test, knn.predict(X_test))\n",
    "\n",
    "print(\"KNN Train set Accuracy: \", '{:.2%}'.format(knn_train_acc))\n",
    "print(\"KNN Test set Accuracy: \", '{:.2%}'.format(knn_test_acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Train set Accuracy:  80.99%\n",
      "Decision Tree Test set Accuracy:  76.92%\n"
     ]
    }
   ],
   "source": [
    "#Decision Tree Accuracy\n",
    "yhat_tree = tree_model.predict(test_X)\n",
    "tree_jaccard = jaccard_score(test_y, yhat_tree, pos_label='PAIDOFF')\n",
    "tree_f1 = f1_score(test_y, yhat_tree, average='weighted')\n",
    "\n",
    "tree_train_acc = metrics.accuracy_score(y_train, tree_model.predict(X_train))\n",
    "tree_test_acc = metrics.accuracy_score(y_test, tree_model.predict(X_test))\n",
    "\n",
    "print(\"Decision Tree Train set Accuracy: \", '{:.2%}'.format(tree_train_acc))\n",
    "print(\"Decision Tree Test set Accuracy: \", '{:.2%}'.format(tree_test_acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Train set Accuracy:  75.21%\n",
      "SVM Tree Test set Accuracy:  76.92%\n"
     ]
    }
   ],
   "source": [
    "#SVM Accuracy\n",
    "yhat_svm = svm_model.predict(test_X)\n",
    "svm_jaccard = jaccard_score(test_y, yhat_svm, pos_label='PAIDOFF')\n",
    "svm_f1 = f1_score(test_y, yhat_svm, average='weighted')\n",
    "\n",
    "svm_train_acc = metrics.accuracy_score(y_train, svm_model.predict(X_train))\n",
    "svm_test_acc = metrics.accuracy_score(y_test, svm_model.predict(X_test))\n",
    "\n",
    "print(\"SVM Train set Accuracy: \", '{:.2%}'.format(svm_train_acc))\n",
    "print(\"SVM Tree Test set Accuracy: \", '{:.2%}'.format(svm_test_acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression set Accuracy:  74.38%\n",
      "Logistic Regression set Accuracy:  76.92%\n"
     ]
    }
   ],
   "source": [
    "#Logistic Regression Accuracy\n",
    "yhat_LR = LR_model.predict(test_X)\n",
    "yhat_prob = LR_model.predict_proba(test_X)\n",
    "LR_jaccard = jaccard_score(test_y, yhat_LR, pos_label='PAIDOFF')\n",
    "LR_f1 = f1_score(test_y, yhat_LR, average='weighted')\n",
    "LR_logloss = log_loss(test_y, yhat_prob)\n",
    "\n",
    "LR_train_acc = metrics.accuracy_score(y_train, LR_model.predict(X_train))\n",
    "LR_test_acc = metrics.accuracy_score(y_test, LR_model.predict(X_test))\n",
    "\n",
    "print(\"Logistic Regression set Accuracy: \", '{:.2%}'.format(LR_train_acc))\n",
    "print(\"Logistic Regression set Accuracy: \", '{:.2%}'.format(LR_test_acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Jaccard Score</th>\n",
       "      <th>F1-score</th>\n",
       "      <th>LogLoss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KNN</td>\n",
       "      <td>0.69</td>\n",
       "      <td>0.66</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Decision Tree</td>\n",
       "      <td>0.7</td>\n",
       "      <td>0.75</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SVM</td>\n",
       "      <td>0.72</td>\n",
       "      <td>0.62</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Logistic Regression</td>\n",
       "      <td>0.74</td>\n",
       "      <td>0.63</td>\n",
       "      <td>0.53</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Model Jaccard Score F1-score LogLoss\n",
       "0                  KNN          0.69     0.66     nan\n",
       "1        Decision Tree           0.7     0.75     nan\n",
       "2                  SVM          0.72     0.62     nan\n",
       "3  Logistic Regression          0.74     0.63    0.53"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "report = pd.DataFrame(data=np.array([[\"KNN\", '{:.2}'.format(knn_jaccard), '{:.2}'.format(knn_f1), np.nan],\n",
    "                                     [\"Decision Tree\", '{:.2}'.format(tree_jaccard), '{:.2}'.format(tree_f1), np.nan],\n",
    "                                     [\"SVM\", '{:.2}'.format(svm_jaccard), '{:.2}'.format(svm_f1), np.nan],\n",
    "                                     [\"Logistic Regression\", '{:.2}'.format(LR_jaccard), '{:.2}'.format(LR_f1), '{:.2}'.format(LR_logloss)]]), \n",
    "                             columns=[\"Model\", \"Jaccard Score\", \"F1-score\", \"LogLoss\"])\n",
    "report"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Report\n",
    "\n",
    "You should be able to report the accuracy of the built model using different evaluation metrics:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| Algorithm          | Jaccard | F1-score | LogLoss |\n",
    "|--------------------|---------|----------|---------|\n",
    "| KNN                | ?       | ?        | NA      |\n",
    "| Decision Tree      | ?       | ?        | NA      |\n",
    "| SVM                | ?       | ?        | NA      |\n",
    "| LogisticRegression | ?       | ?        | ?       |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<h2>Want to learn more?</h2>\n",
    "\n",
    "IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href=\"http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01\">SPSS Modeler</a>\n",
    "\n",
    "Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href=\"https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01\">Watson Studio</a>\n",
    "\n",
    "<h3>Thanks for completing this lesson!</h3>\n",
    "\n",
    "<h4>Author:  <a href=\"https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01\">Saeed Aghabozorgi</a></h4>\n",
    "<p><a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>\n",
    "\n",
    "<hr>\n",
    "\n",
    "## Change Log\n",
    "\n",
    "| Date (YYYY-MM-DD) | Version | Changed By | Change Description |\n",
    "|---|---|---|---|\n",
    "| 2020-10-27 | 2.1 | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |\n",
    "| 2020-08-27 | 2.0 | Malika Singla | Added lab to GitLab |\n",
    "\n",
    "<hr>\n",
    "\n",
    "## <h3 align=\"center\"> © IBM Corporation 2020. All rights reserved. <h3/>\n",
    "\n",
    "<p>\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.8",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
