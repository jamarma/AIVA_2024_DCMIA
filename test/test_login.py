import unittest


from src.app.auth.models import User


class TestLogin(unittest.TestCase):

    def setUp(self):
        self.user = User()

    def test_login_successful(self):
        self.assertTrue(self.user.verify_password("password1"))

    def test_login_failed_incorrect_password(self):
        self.assertFalse(self.user.verify_password("wrong_password"))


if __name__ == '__main__':
    unittest.main()