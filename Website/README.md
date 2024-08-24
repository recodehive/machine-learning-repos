# Machine Learning Repos Contributing Guidelines

## Prerequisites ⚠️

- Open Source Etiquette: If you've never contributed to an open source project before, have a read of [Basic etiquette](https://developer.mozilla.org/en-US/docs/MDN/Community/Open_source_etiquette) for open source projects.

- Basic familiarity with Git and GitHub: If you are also new to these tools, visit [GitHub for complete beginners](https://developer.mozilla.org/en-US/docs/MDN/Contribute/GitHub_beginners) for a comprehensive introduction to them.

---

### Setup guidelines 🪜

**Follow these steps to setup Machine-Learning-Repos on your local machine 👇**

- [Fork](https://github.com/recodehive/machine-learning-repos/fork) the repository
- Clone the forked repository in your local system.
  
  ```bash
   https://github.com/recodehive/machine-learning-repos.git
  ```
 - Navigate to the [Website](https://github.com/recodehive/machine-learning-repos/tree/main/Website) folder if you want to contribute to our website.
   ```bash
    cd Website
   ```
 - Now install dependency
   ```bash
    cd server
    npm install
   ```
- Setup env file
  ```
  1. Create .env in the server folder
  2. Store your Github PAT Token
  GITHUB_TOKEN = YOUR_GITHUB_TOKEN
  ```
  - Run the Server
    ```bash
     cd server
     node server.js
    ```
  - `Open http://localhost:3000 with your browser to see the result.`
  
 - Create a new branch for your feature.
   ```bash
    git checkout -b <your_branch_name>
   ```
 - Perform your desired changes to the code base.
 - Track and stage your changes.
   ```bash
    # Track the changes
     git status

    # Add changes to Index
     git add .
   ```
- Commit your changes.
  ```bash
  git commit -m "your_commit_message"
  ```
- Push your committed changes to the remote repo.
  ```bash
  git push origin <your_branch_name>
  ```
- Go to your forked repository on GitHub and click on `Compare & pull request`.
- Add an appropriate title and description to your pull request explaining your changes and efforts done.
- Click on `Create pull request`.
- Congrats! 🥳 You've made your first pull request to this project repo.
- Wait for your pull request to be reviewed and if required suggestions would be provided to improve it.
- Celebrate 🥳 your success after your pull request is merged successfully.


## ✅ Guidelines for Good Commit Messages 
We follow a standardized commit message format using Commitlint to ensure consistency and clarity in our commit history. Each commit message should adhere to the following guidelines:

1. **Be Concise and Descriptive**: Summarize the change in a way that’s easy to understand at a glance.
2. **Use the Imperative Mood**: Write as if giving a command (e.g., `Add`, `Fix`, `Update`), which is a convention in many projects.
3. **Include Context**: Provide context or reason for the change if it’s not immediately obvious from the summary.
4. **Reference Issues and Pull Requests**: Include `issue numbers` or PR references if the commit addresses them.
5. **Issue reference** (Optional): Include the issue number associated with the commit (e.g., `#123`).

## 📝 Commit Message Examples ✅
### Enhancing Documentation
- `Improve - readability of <document> tutorial`
- `Add - examples to ML documentation`
- `Enhance - troubleshooting section in Prometheus guide`

### General Maintenance
- `Refactor - README for better clarity`
- `Reorganize repository structure for easier navigation`
- `Remove - outdated tools from recommendations`

# ❌ Examples of Invalid Commit Messages

- `Added new stuff`
- `Fixed a bug`
- `Updated code`
- `auth feature update`
- `chore: fixed some stuff`

## Commit Example with Commitlint

```bash
git commit -m "feat(auth): Implement user signup process (#789)"
```

---

- If something is missing here, or you feel something is not well described, please [raise an issue](https://github.com/recodehive/machine-learning-repos/issues).

