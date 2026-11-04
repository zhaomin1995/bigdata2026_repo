import torch
import re

from transformers import EncoderDecoderModel
from transformers import AutoTokenizer


class TextFilter:
    """Remove tweets based on the tweet text.

    If it returns True, then remove the current tweet.
    If it returns False, then keep the current tweet.
    Note that the type of temporal expression could be one of TIME, DATE, SET, DURATION.
    Since our project focuses on duration, I only check the temporal expression.
    Around 11% of all tweets contain duration expression.

    References
    ----------
    https://www.aaai.org/Papers/Symposia/Spring/2003/SS-03-07/SS03-07-005.pdf
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1084.2779&rep=rep1&type=pdf
    https://arxiv.org/abs/2109.14927

    Examples
    --------
    >>> from utils.tweet_filter.text_filter import TextFilter
    >>> text_filter = TextFilter()
    >>> tweet_text = tweet['full_text']
    >>> text_filter.check_duration_expression(tweet_text)
    True
    """

    def __init__(self):
        self.temporal_types = ["type=\"DURATION\""]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
        self.model = EncoderDecoderModel.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
        self.model = self.model.to(self.device)

    def check_duration_expression(self, tweet_text):
        """If tweet text contain duration expression, then remove it.

        Use transformer-based temporal tagger to find temporal expression in the given text.
        If tweet text contain duration expression, such as 5 mins and four years, remove it.

        Args:
            The tweet text.
        Returns:
            True: this tweet should be removed.
            False: this tweet can be kept.
        """
        # remove URL(s)
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        model_inputs = self.tokenizer(tweet_text, truncation=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**model_inputs)
        decoded_preds = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        if any(temporal_type in decoded_preds[0] for temporal_type in self.temporal_types):
            return True
        else:
            return False
