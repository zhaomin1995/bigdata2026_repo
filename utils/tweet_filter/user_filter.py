import os

from pathlib import Path


class OrganizationUserIdentifier:
    """Use user information to identify if the user is organization tweet or not.

    If it returns True, it means the tweet is most likely published by an organization.
    If it returns False, it means tha tweet is most likely published by a personal user.

    References
    ----------
    https://aclanthology.org/W18-1108/

    Examples
    --------
    >>> from utils.tweet_filter.user_filter import OrganizationUserIdentifier
    >>> user_identifier = OrganizationUserIdentifier()
    >>> user_info = tweet['user']
    >>> user_identifier.ratio_of_follower_friend(user_info)
    True
    >>> user_identifier.num_of_history_tweet(user_info)
    False
    """

    def __init__(self, max_follower_friend_ratio=50, max_num_tweets=50000):
        self.max_follower_friend_ratio = max_follower_friend_ratio
        self.max_num_tweets = max_num_tweets

        current_path = Path(__file__).parent.resolve()
        self.keyword_filepath = os.path.join(current_path, 'username_keyword')

    def ratio_of_follower_friend(self, user_info):
        """Discard tweets based on the ratio of the number of followers over friends.

        If # of followers / # of friends is over the predefined threshold, discard it.

        Args:
            user metadata

        Returns:
            True: we discard this tweet
            False: we can keep this tweet
        """
        followers_count = user_info['followers_count']
        friends_count = user_info['friends_count']
        if followers_count / max(1, friends_count) > self.max_follower_friend_ratio:
            return True
        else:
            return False

    def num_of_history_tweet(self, user_info):
        """Discard tweets based on the number of tweets the user has published.

        If the user has published more than n tweets, discard it,
        where n is pre-defined parameter.

        Args:
            user metadata

        Returns:
            True: we discard this tweet
            False: we can keep this tweet
        """
        num_tweets = user_info['statuses_count']
        if num_tweets > self.max_num_tweets:
            return True
        else:
            return False

    def keyword_filter(self, user_info):
        """Discard tweets based on if username / screen_name contains specific keyword.

        Read the keyword list from the external keyword file.
        If username / screen_name contains any keyword in that list, discard it.

        Args:
            user metadata
        Returns:
            True: we discard this tweet
            False: we can keep this tweet
        """
        keep = True
        with open(self.keyword_filepath, 'r') as keyword_file:
            keywords = keyword_file.read().split('\n')[:-1]
        for keyword in keywords:
            if keyword in user_info['screen_name'].lower() or keyword in user_info['name'].lower().replace(' ', ''):
                keep = False
                break
        if not keep:
            return True
        else:
            return False
