from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from src.Models.load_data import *
from src.constants import TOTAL_NUM_HEROES
from src.Models.preprocess_data.synergy_and_counter import get_match_syncoun, get_match_wr
from src.Models.preprocess_data.add_wr_pick_time import get_wr_for_team
from src.Models.preprocess_data.add_wr_pick_time import process_match_duration
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import os
import time
import json


FILE_DIR = os.path.dirname(os.path.realpath(__file__))

# Function to load all the images in the image directory
IMG_DIR = os.path.join(FILE_DIR, 'Heroes Portraits')
def get_heroes_img():
    for file in os.listdir(IMG_DIR):
        hero_name = file[:-4]
        heroes_dict[hero_name].append(file)

# Load hero data and divide them into list of core and support heroes
def get_heroes():
    json_file = None
    try:
        with open('E:\Disertation\dota2ml\src\Models/heroes.json') as f:
            json_file = json.load(f)
    except:
        print("Could not open heroes.json file")
        exit()

    heroes_list = []
    heroes_dict = {}
    core_list = []
    support_list = []

    for hero in json_file:
        heroes_list.append(hero['localized_name'])
        heroes_dict[hero['localized_name']] = [hero['id']]
        if "Core" in hero["roles"]:
            core_list.append(hero['id'])
        if "Support" in hero["roles"]:
            support_list.append(hero['id'])

    return heroes_list, heroes_dict, core_list, support_list


class RecommendGUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        # Initialise the lists of chosen heroes
        self.radiant_hero = [None] * 5
        self.dire_hero = [None] * 5
        # Initialise the current chosen team: Radiant team = 1, Dire team = -1
        self.team = 1
        self.style = ttk.Style(master)
        self.style.theme_use('clam')
        self.style.configure('switch.TButton', font='helvetica 10')
        self.style.configure('suggest.TLabelframe')

        # Empty images for buttons
        self.empty_image = self.load_empty_image()

        self.switch_team_button = ttk.Button(master, text='Radiant', style='switch.TButton',
                                             command=lambda: self.switch_team(), width=15)
        self.switch_team_button.grid(row=2, column=2, sticky=W, pady=(20, 0))

        # Create frames for teams
        self.radiant_frame = Frame(master, highlightbackground="red", highlightthickness=2)
        self.radiant_frame.grid(row=4, column=25, columnspan=20, rowspan=7)
        self.dire_frame = Frame(master, highlightbackground="red", highlightthickness=2)
        self.dire_frame.grid(row=4, column=55, columnspan=20, rowspan=7)

        # Initialise win rates and syn coun values
        self.win_rates = [0, 0]
        self.radiant_syncoun_values = [0, 0]
        self.dire_syncoun_values = [0, 0]

        # Draw these labels
        self.radiant_wr = Label(master, text='Radiant Winrate: ' + str(self.win_rates[0]))
        self.radiant_wr.grid(row=12, column=25, padx=(16), sticky=N+W)
        self.dire_wr = Label(master, text='Dire Winrate: ' + str(self.win_rates[1]))
        self.dire_wr.grid(row=12, column=55, padx=(16), sticky=N+W)

        self.radiant_syn = Label(master, text='Radiant Synergy: ' + str(self.radiant_syncoun_values[0]))
        self.radiant_syn.grid(row=12, column=25, padx=(16), pady=(25, 0), sticky=N+W)
        self.radiant_coun = Label(master, text='Radiant Counter: ' + str(self.radiant_syncoun_values[1]))
        self.radiant_coun.grid(row=12, column=25, padx=(16), pady=(50, 0), sticky=N+W)

        self.dire_syn = Label(master, text='Dire Synergy: ' + str(self.dire_syncoun_values[0]))
        self.dire_syn.grid(row=12, column=55, padx=(16), pady=(25, 0), sticky=N+W)
        self.dire_coun = Label(master, text='Dire Counter: ' + str(self.dire_syncoun_values[1]))
        self.dire_coun.grid(row=12, column=55, padx=(16), pady=(50, 0), sticky=N+W)

        # Initialise the suggested heroes
        self.radiant_core_suggestions = []
        self.radiant_support_suggestions = []
        self.dire_core_suggestions = []
        self.dire_support_suggestions = []

        # Setup the frames
        self.radiant_core_frame = ttk.LabelFrame(master, width=250, height=650, text='Radiant Suggestions - Core',
                                                 style='suggest.TLabelframe', borderwidth=4, relief='ridge')
        self.radiant_core_frame.grid_propagate(0)
        self.radiant_core_frame.grid(row=13, rowspan=25, column=25, columnspan=10, padx=(25))

        self.radiant_support_frame = ttk.LabelFrame(master, width=250, height=650, text='Radiant Suggestions - Support',
                                                    style='suggest.TLabelframe', borderwidth=4, relief='ridge')
        self.radiant_support_frame.grid_propagate(0)
        self.radiant_support_frame.grid(row=13, rowspan=25, column=35, columnspan=10, sticky=W)

        self.dire_core_frame = ttk.LabelFrame(master, width=250, height=650, text='Dire Suggestions - Core',
                                              style='suggest.TLabelframe', borderwidth=4, relief='ridge')
        self.dire_core_frame.grid_propagate(0)
        self.dire_core_frame.grid(row=13, rowspan=25, column=55, columnspan=10, padx=(25))

        self.dire_support_frame = ttk.LabelFrame(master, width=250, height=650, text='Dire Suggestions - Support',
                                                 style='suggest.TLabelframe', borderwidth=4, relief='ridge')
        self.dire_support_frame.grid_propagate(0)
        self.dire_support_frame.grid(row=13, rowspan=25, column=65, columnspan=10, sticky=W)

        # Load models
        self.models = {}
        models = ['TimePickAugLog', 'TimePickAugXGB', 'TimePickAugMLP']
        self.current_model = StringVar(master)
        self.current_model.set(models[0])
        for i in models:
            self.models[i] = load_model(i)

        # The button for changing models
        model_dropdown = ttk.OptionMenu(master, self.current_model, models[0], *models,
                                        command=self.update_winrates, style='switch.TButton')
        model_dropdown.config(width=15)
        model_dropdown.grid(row=2, column=5, pady=(20, 0))

        # Initiliase the time scale bar
        self.match_time = DoubleVar()
        self.match_time.set(30.0)

        self.time_scale_label = Label(master, text='Duration: ' + str(self.match_time.get()))
        self.time_scale_label.grid(row=3, column=9, columnspan=2, sticky=S + W, pady=(0,15))
        self.match_time_scale = ttk.Scale(master, length=225, from_=15.0, to_=50, variable=self.match_time,
                                          command=self.update_scale)
        self.match_time_scale.grid(row=3,column=9, columnspan=2, sticky=N, pady=(27, 0))
        show_time_prediction = ttk.Button(master, text='Time Prediction', style='switch.TButton',
                                         command = self.show_graph, width=15)
        show_time_prediction.grid(row=3, column=2, sticky=W, pady=(10, 0))

        # Button to switch sort mode: 1: sort by syncoun, -1: sort by wr
        self.sort_mode = 1
        self.sort_button = ttk.Button(master, text='Sort by SynCoun', style='switch.TButton',
                                          command=self.switch_sort_mode, width=15)
        self.sort_button.grid(row=3, column=5, sticky=W, pady=(10, 0))

        # Button to switch suggestion mode: 0: Synergy, 1: Counter, 2: All
        self.suggestion_mode = 0
        self.swich_suggest_button = ttk.Button(master, text='Synergy Only', style='switch.TButton',
                                          command=self.switch_suggest_mode, width=15)
        self.swich_suggest_button.grid(row=2, column=9, sticky=W, pady=(20, 0))

        # Button to display hotkeys
        show_hotkeys = ttk.Button(master, text='Show Hotkeys', style='switch.TButton',
                                          command=self.show_hotkeys, width=15)
        show_hotkeys.grid(row=2, column=10, sticky=W, pady=(20, 0), padx=(5,0))

        # Binding the hotkeys
        master.bind('<F1>', self.switch_team)
        master.bind('<F2>', self.switch_sort_mode)
        master.bind('<F3>', self.switch_suggest_mode)
        master.bind('<Escape>', lambda x: exit())

        self.init_frames()

    def init_frames(self):
        self.master.title("Heroes Recommender")

        # Initialise the heroes frame where heroes are chosen
        heroes_frame = Frame(self.master, highlightbackground="green", highlightthickness=2)
        heroes_frame.grid(row=10, rowspan=30, column=0, columnspan=20)

        # Make button for each hero
        for hero_name in heroes_dict:
            hero_id = heroes_dict[hero_name][0]
            self.make_button(heroes_frame, hero_name, hero_id, resize=2.8, row=int(hero_id / 6),
                             column=int(hero_id % 6))

        for i in range(5):
            # Create empty buttons for Radiant chosen heroes
            self.team_button(self.radiant_frame, i, team=1)

        for i in range(5):
            # Create empty buttons for Dire chosen heroes
            self.team_button(self.dire_frame, i, team=-1)

    # Function to create button to choose heroes
    def make_button(self, frame, hero_name, hero_id, resize, row, column, team=None, padx=0, pady=0):
        b = Button(frame, command=lambda: self.choose_hero(hero_name, hero_id, team))
        image = self.load_image(hero_name, resize=resize)
        b.config(image=image)
        b.grid(row=row, column=column, padx=padx, pady=pady)
        b.image = image

    # Function to create button that shows chosen heroes
    # Clicking on these buttons remove the heroes from the chosen heroes list
    def team_button(self, frame, column, hero_name=None, hero_id=0, team=1):
        if hero_name is not None:
            b = Button(frame, command=lambda: self.remove_hero(b, hero_id, team))
            image = self.load_image(hero_name, resize=1.7)
            b.config(image=image)
            b.grid(row=0, column=column)
            b.image = image
        else:
            b = Button(frame)
            b.config(image=self.empty_image)
            b.grid(row=0, column=column)

    # Function to remove hero from chosen heroes list
    def remove_hero(self, b, hero_id, team):
        b.config(image=self.empty_image, command=lambda: None)
        if team == 1:
            self.radiant_hero[self.radiant_hero.index(hero_id)] = None
        else:
            self.dire_hero[self.dire_hero.index(hero_id)] = None

        self.update_winrates()

    # Function handling what happens when choosing heroes
    def choose_hero(self, hero_name, hero_id, team=None):
        if team is None:
            team = self.team
        # Check if the hero is already chosen, which team is currently chosen and whether the team is full
        if hero_id not in self.radiant_hero and hero_id not in self.dire_hero:
            if team == 1:
                if self.radiant_hero.count(None) > 0:
                    index = self.radiant_hero.index(None)
                    self.team_button(self.radiant_frame, index, hero_name, hero_id, team=1)
                    self.radiant_hero[index] = hero_id
            else:
                if self.dire_hero.count(None) > 0:
                    index = self.dire_hero.index(None)
                    self.team_button(self.dire_frame, index, hero_name, hero_id, team=-1)
                    self.dire_hero[index] = hero_id

            self.update_winrates()

    # Function to switch currently chosen team
    def switch_team(self, _event=None):
        texts = {1: 'Radiant', -1: 'Dire'}
        self.team *= -1
        self.switch_team_button.config(text=texts[self.team])

    # Function to switch sort mode
    def switch_sort_mode(self, _event=None):
        texts = {1: 'SynCoun', -1: 'Winrate'}
        self.sort_mode *= -1
        self.sort_button.config(text='Sort by ' + texts[self.sort_mode])
        self.update_winrates()

    # Function to switch suggestion mode
    def switch_suggest_mode(self, _event=None):
        texts = {0: 'Synergy Only', 1: 'Counter Only', 2: 'Both'}
        if self.suggestion_mode == 2:
            self.suggestion_mode = 0
        else:
            self.suggestion_mode += 1
        self.swich_suggest_button.config(text=texts[self.suggestion_mode])
        self.update_winrates()

    def show_hotkeys(self, _event=None):
        messagebox.showinfo('Hotkeys Info', "F1: Switch Team \nF2: Switch Sort Mode \nF3: Switch Suggestion Mode\n"
                                            "Esc: Exit")

    # Function to handle new predictions
    def predict(self, radiant_team, dire_team, predict_time = False):
        # Initiliase the query using currently chosen heroes
        query = self.generate_query(radiant_team, dire_team)

        # Get the syn coun values and avg win rates for both teams
        rad_syn_adv, rad_coun_adv, dire_syn_adv, dire_coun_adv = get_match_syncoun(radiant_team, dire_team,
                                                                                   syn_matrix, coun_matrix)
        rad_avg_wr, dire_avg_wr = get_match_wr(radiant_team, dire_team, wr_matrix)
        query[-3] = rad_syn_adv - dire_syn_adv
        query[-2] = rad_coun_adv
        query[-1] = rad_avg_wr - dire_avg_wr
        syncoun_values = [rad_syn_adv, rad_coun_adv, dire_syn_adv, dire_coun_adv]
        syncoun_values = [x * 100 for x in syncoun_values]

        # Calculate pick order values
        if 'Pick' in self.current_model.get():
            rad_pick_wr = get_wr_for_team(radiant_team, pick_matrix)
            dire_pick_wr = get_wr_for_team(dire_team, pick_matrix)
            query[-5] = rad_pick_wr
            query[-4] = dire_pick_wr

        # If predict_time is True, i.e. the function is called for the time prediction graph
        if predict_time:
            rad_time_wr = [None] * 5
            dire_time_wr = [None] * 5

            for i in range(5):
                query[-7], query[-6] = process_match_duration(radiant_team, dire_team, time_matrix, index=i)
                result = self.models[self.current_model.get()].predict_proba(query.reshape(1, -1))[
                       0].tolist()
                rad_time_wr[i] = result[1]
                dire_time_wr[i] = result[0]

            return rad_time_wr, dire_time_wr

        # Calculate match duration values
        elif 'Time' in self.current_model.get():
            query[-7], query[-6] = process_match_duration(radiant_team, dire_team, time_matrix, duration=self.match_time.get())

        return self.models[self.current_model.get()].predict_proba(query.reshape(1, -1))[
                   0].tolist(), syncoun_values

    # Function to handle generating query given the input teams
    def generate_query(self, radiant_team, dire_team):
        extra_columns = 0
        if 'Aug' in self.current_model.get():
            extra_columns += 3
        if 'Pick' in self.current_model.get():
            extra_columns += 2
        if 'Time' in self.current_model.get():
            extra_columns += 2
        query = np.zeros((TOTAL_NUM_HEROES + extra_columns, 1))

        for hero in radiant_team:
            query[hero] = 1
        for hero in dire_team:
            query[hero] = -1

        return query

    # Function to handle update win rates whenever an action occurs
    def update_winrates(self, _event=None):
        rad_hero_count = 5 - self.radiant_hero.count(None)
        dire_hero_count = 5 - self.dire_hero.count(None)
        current_model = self.current_model.get()

        radiant_team = [x for x in self.radiant_hero if x is not None]
        dire_team = [x for x in self.dire_hero if x is not None]

        # Only do prediction of number of heroes in both teams are more than 2
        min_count = 2 if 'Aug' in current_model else 1
        if rad_hero_count >= min_count and dire_hero_count >= min_count:
            self.win_rates, syncoun_values = self.predict(radiant_team, dire_team)
            self.win_rates = np.around(self.win_rates, 2)
            self.radiant_syncoun_values = np.around(syncoun_values[:2], 2)
            self.dire_syncoun_values = np.around(syncoun_values[2:], 2)

            # Call the function to get suggestion if the team has fewer than 5 heroes
            if rad_hero_count < 5 or dire_hero_count < 5:
                self.get_suggestion(radiant_team, dire_team)
            if rad_hero_count == 5 and dire_hero_count == 5:
                self.update_texts(clean=True)
            # Clearing the suggestion frames whenever each team has full 5 heroes
            elif rad_hero_count == 5:
                self.update_texts(clean_rad=True)
            elif dire_hero_count == 5:
                self.update_texts(clean_dire=True)
            else:
                self.update_texts()
        else:
            self.reset_values()


    # Function to handle getting suggestions
    def get_suggestion(self, radiant_team, dire_team):
        # start_time = time.time()
        available_pool = list(range(0, 115))
        rad_suggestions = []
        dire_suggestions = []
        # Get an available pools of unchosen heroes
        for i in radiant_team + dire_team:
            available_pool.remove(i)

        # For each hero, get the win rate and syn coun values of the new feature vector
        # with the hero added in each team.
        for try_hero in available_pool:
            temp_rad_wr, temp_rad_syncoun = self.predict(radiant_team + [try_hero], dire_team)
            temp_dire_wr, temp_dire_syncoun = self.predict(radiant_team, dire_team + [try_hero])

            if self.suggestion_mode == 0:
                temp_rad_syncoun = temp_rad_syncoun[0]
                temp_dire_syncoun = temp_dire_syncoun[2]
            elif self.suggestion_mode == 1:
                temp_rad_syncoun = temp_rad_syncoun[1]
                temp_dire_syncoun = temp_dire_syncoun[3]
            else:
                temp_rad_syncoun = temp_rad_syncoun[0] + temp_rad_syncoun[1]
                temp_dire_syncoun = temp_dire_syncoun[2] + temp_dire_syncoun[3]

            rad_suggestions.append((try_hero, temp_rad_wr[1], temp_rad_syncoun))
            dire_suggestions.append((try_hero, temp_dire_wr[0], temp_dire_syncoun))

        # Get the current sort mode
        if self.sort_mode == 1:
            rad_suggestions.sort(key=lambda x: x[2], reverse=True)
            dire_suggestions.sort(key=lambda x: x[2], reverse=True)
        else:
            rad_suggestions.sort(key=lambda x: x[1], reverse=True)
            dire_suggestions.sort(key=lambda x: x[1], reverse=True)

        # Display suggestions
        num_suggestions = 8
        rad_core_suggestions, rad_support_suggestions, dire_core_suggestions, dire_support_suggestions = [], [], [], []
        for i, (suggest_type, suggest, list_type) in enumerate([(rad_core_suggestions, rad_suggestions, core_list),
                                                                (rad_support_suggestions, rad_suggestions, support_list),
                                                                (dire_core_suggestions, dire_suggestions, core_list),
                                                                (dire_support_suggestions, dire_suggestions,support_list)]):
            # Divide the suggestions into core and support heroes
            while len(suggest_type) < num_suggestions:  # rad_core_suggestions
                for hero in suggest:  # rad_suggestions
                    if hero[0] in list_type:  # core_list
                        suggest_type.append(hero)

        # Get the top 8 heroes in each category
        rad_core_suggestions = rad_core_suggestions[:num_suggestions]
        dire_core_suggestions = dire_core_suggestions[:num_suggestions]
        rad_support_suggestions = rad_support_suggestions[:num_suggestions]
        dire_support_suggestions = dire_support_suggestions[:num_suggestions]

        # Set up a new list where which hero, syn coun values, win rates information are gathered
        self.radiant_core_suggestions = [(heroes_list[rad_core_suggestions[i][0]],
                                        rad_core_suggestions[i][0],
                                        rad_core_suggestions[i][2],
                                        rad_core_suggestions[i][1]) for i in
                                        range(len(rad_core_suggestions))]
        self.dire_core_suggestions = [(heroes_list[dire_core_suggestions[i][0]],
                                       dire_core_suggestions[i][0], dire_core_suggestions[i][2],
                                       dire_core_suggestions[i][1]) for i in
                                      range(len(dire_core_suggestions))]
        self.radiant_support_suggestions = [(heroes_list[rad_support_suggestions[i][0]],
                                             rad_support_suggestions[i][0],
                                             rad_support_suggestions[i][2],
                                             rad_support_suggestions[i][1])for i
                                            in range(len(rad_support_suggestions))]
        self.dire_support_suggestions = [(heroes_list[dire_support_suggestions[i][0]],
                                          dire_support_suggestions[i][0],
                                          dire_support_suggestions[i][2],
                                          dire_support_suggestions[i][1]) for i in
                                         range(len(dire_support_suggestions))]
        # print("It took %s for %s" % (time.time() - start_time, self.current_model.get()))

    def reset_values(self):
        self.win_rates = [0, 0]
        self.radiant_syncoun_values = [0, 0]
        self.dire_syncoun_values = [0, 0]
        self.update_texts(clean=True)

    # Update the displayed texts
    def update_texts(self, clean=False, clean_rad=False, clean_dire=False):
        self.radiant_wr.config(text='Radiant Winrate: ' + str(self.win_rates[1]))
        self.dire_wr.config(text='Dire Winrate: ' + str(self.win_rates[0]))
        self.radiant_syn.config(text='Radiant Synergy: ' + str(self.radiant_syncoun_values[0]))
        self.radiant_coun.config(text='Radiant Counter: ' + str(self.radiant_syncoun_values[1]))
        self.dire_syn.config(text='Dire Synergy: ' + str(self.dire_syncoun_values[0]))
        self.dire_coun.config(text='Dire Counter: ' + str(self.dire_syncoun_values[1]))

        # Clean the previous suggestions to add new ones
        self.clean_suggestions()

        # Display the suggestions
        if not clean:
            do_rad = [(self.radiant_core_frame, self.radiant_core_suggestions, 1),
                      (self.radiant_support_frame,self.radiant_support_suggestions, 1)]
            do_dire = [(self.dire_core_frame, self.dire_core_suggestions, -1),
                       (self.dire_support_frame, self.dire_support_suggestions,-1)]
            if clean_rad:
                do_all = do_dire
            elif clean_dire:
                do_all = do_rad
            else:
                do_all = do_rad+do_dire

            for i, (frame, suggestions, team) in enumerate(do_all):
                for suggest in suggestions:
                    hero_name = suggest[0]
                    hero_id = suggest[1]
                    add_syncoun_values = np.around(suggest[2], 2)
                    temp_wr = np.around(suggest[3], 2)
                    # Creating new heroes button for suggestions
                    self.make_button(frame, hero_name, hero_id, resize=2, row=suggestions.index(suggest), column=0,
                                     team=team, pady=5)
                    self.make_labels(frame, hero_name, add_syncoun_values, temp_wr,
                                     row=suggestions.index(suggest), column=2,
                                     padx=10)

    # Function handling cleaning suggestions
    def clean_suggestions(self):
        rad = [self.radiant_core_frame, self.radiant_support_frame]
        dire = [self.dire_support_frame, self.dire_core_frame]

        for frame in rad + dire:
            for child in frame.winfo_children():
                child.destroy()

    # Labels to display suggested heroes syn coun values and win rates
    def make_labels(self, frame, hero_name, syncoun_values, temp_wr, row, column, padx, ):
        l1 = ttk.Label(frame, text=hero_name)
        l1.grid(row=row, column=column, padx=padx, sticky=W + N, pady=(5, 0))

        operator = '+ ' if syncoun_values > 0 else '- '
        l2 = ttk.Label(frame, text='SynCoun: ' + operator + str(abs(syncoun_values)))
        l2.grid(row=row, column=column, padx=padx, sticky=W + S, pady=(0, 10))

        l3 = ttk.Label(frame, text='Winrate: ' + str(int(temp_wr*100)) + '%')
        l3.grid(row=row, column=column, padx=padx, sticky=W + S, pady=(0, 25))

    # Function to display win rate prediction graph
    def show_graph(self, _event = None):
        if not 'Time' in self.current_model.get():
            messagebox.showerror("Error", "Can only display time prediction with a model with prefix 'Time'")
            return

        rad_hero_count = 5 - self.radiant_hero.count(None)
        dire_hero_count = 5 - self.dire_hero.count(None)

        if rad_hero_count < 2 and dire_hero_count < 2:
            messagebox.showerror("Error", "Can only display time prediction when each team has at least two heroes")
            return

        radiant_team = [x for x in self.radiant_hero if x is not None]
        dire_team = [x for x in self.dire_hero if x is not None]

        rad_time_wr, dire_time_wr = self.predict(radiant_team, dire_team, predict_time=True)
        window = Toplevel(root)

        f = Figure(figsize=(7, 5), dpi=100)
        plt = f.add_subplot(111)
        index = [15, 20, 30, 40, 50]

        plt.plot(index, rad_time_wr, label='Radiant Winrate')
        plt.plot(index, dire_time_wr, label='Dire Winrate')
        plt.set_xlabel('Duration of match')
        plt.set_ylabel('Winrate')
        plt.set_title('Winrate of each team at different times')
        plt.legend()

        canvas = FigureCanvasTkAgg(f, window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        window.bind('<Escape>', lambda x: window.destroy())

    # Update the time scale whenever user changes it
    def update_scale(self, foo=None):
        self.time_scale_label.config(text= 'Duration: ' + str(np.around(self.match_time.get(),0)))
        self.update_winrates()

    def load_image(self, hero_name, resize):
        hero_img = os.path.join(IMG_DIR, heroes_dict[hero_name][1])
        photo = Image.open(hero_img)
        photo = photo.resize((int(205 / resize), int(115 / resize)))
        image = ImageTk.PhotoImage(photo)
        return image

    def load_empty_image(self):
        empty_image = os.path.join(FILE_DIR, 'Empty.jpg')
        photo = Image.open(empty_image)
        photo = photo.resize((int(205 / 1.7), int(115 / 1.7)))
        image = ImageTk.PhotoImage(photo)
        return image

    def client_exit(self):
        exit()


class ChooseModels:
    def __init__(self, master):
        Frame.__init__(master)

if __name__ == '__main__':
    # Get the list of heroes and list of core and support heroes
    heroes_list, heroes_dict, core_list, support_list = get_heroes()
    get_heroes_img()
    # Load features data
    syn_matrix, coun_matrix = load_data('SynCoun')
    wr_matrix = load_data('Winrate')
    time_matrix, pick_matrix = load_data('TimePickWr')

    root = Tk()

    root.wm_state('zoomed')
    rows, columns = 0, 0
    while rows < 40:
        root.grid_rowconfigure(rows, weight=20)
        rows += 1

    while columns < 80:
        root.grid_columnconfigure(columns, weight=20)
        columns += 1

    app = RecommendGUI(root)

    root.mainloop()
