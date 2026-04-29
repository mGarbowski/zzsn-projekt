
## Dostęp do repo

1. Z poziomu anteny `ssh-keygen` i zrzucasz klucz np. do `~/.ssh/id_zzsn_github`:

2. `~/.ssh/config`:
```

Host github.com
        User git
        IdentityFile ~/.ssh/id_zzsn_github

```

3. Dodajesz deploy key przez settings w repo githubowym

4. Klonujesz i powinno śmigać
