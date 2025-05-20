#include "card_detection.hpp"

vector<Train_ranks> load_ranks(const string &filepath)
{
	vector<Train_ranks> train_ranks;
	vector<string> rank_names = {"Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
								 "Eight", "Nine", "Ten", "Jack", "Queen", "King"};

	for (const auto &name : rank_names)
	{
		Train_ranks tr;
		tr.name = name;
		tr.img = imread(filepath + "/" + name + ".jpg", IMREAD_GRAYSCALE);
		if (!tr.img.empty())
			train_ranks.push_back(tr);
	}
	return train_ranks;
}

// Load training suit images
vector<Train_suits> load_suits(const string &filepath)
{
	vector<Train_suits> train_suits;
	vector<string> suit_names = {"Spades", "Diamonds", "Clubs", "Hearts"};

	for (const auto &name : suit_names)
	{
		Train_suits ts;
		ts.name = name;
		ts.img = imread(filepath + "/" + name + ".jpg", IMREAD_GRAYSCALE);
		if (!ts.img.empty())
			train_suits.push_back(ts);
	}
	return train_suits;
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		cout << "Usage: " << argv[0] << " <image_path> <train_images_directory>" << endl;
		return -1;
	}

	string image_path = argv[1];
	string train_dir = argv[2];

	Mat image = imread(image_path);
	if (image.empty())
	{
		cout << "Could not load image." << endl;
		return -1;
	}

	vector<Train_ranks> train_ranks = load_ranks(train_dir);
	vector<Train_suits> train_suits = load_suits(train_dir);

	Mat thresh = preprocess_image(image);
	vector<vector<Point>> cnts_sort;
	vector<int> cnt_is_card;
	find_cards(thresh, cnts_sort, cnt_is_card);

	for (size_t i = 0; i < cnts_sort.size(); i++)
	{
		if (cnt_is_card[i] == 1)
		{
			Query_card qCard = preprocess_card(cnts_sort[i], image);
			match_card(qCard, train_ranks, train_suits);
			draw_results(image, qCard);
		}
	}

	resize(image, image, Size(800, 600));
	imshow("Detected Cards", image);
	waitKey(0);
	return 0;
}