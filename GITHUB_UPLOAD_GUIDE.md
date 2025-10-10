# GitHub Upload Guide - FATES-Emulator

## 🚀 Step-by-Step Instructions to Upload to GitHub

---

## Prerequisites

✅ You have:
- A GitHub account
- Git installed on your system
- SSH keys set up (or will use HTTPS)

---

## Step 1: Initialize Git Repository (On Your Machine)

```bash
# Navigate to your repository
cd /qfs/people/lili400/compy/project/NGT/manaus/script/3.7_git_repo

# Initialize git (if not already done)
git init

# Add all files
git add .

# Check what will be committed
git status

# Make first commit
git commit -m "Initial commit: FATES-Emulator v0.1.0

- Complete ML framework with FLAML AutoML
- Parameter sampling with ecological constraints
- SHAP interpretability analysis
- Multi-objective calibration
- No hardcoded paths
- Complete documentation
- Example workflows

Based on Li et al. (2023) GMD: https://doi.org/10.5194/gmd-16-4017-2023"
```

---

## Step 2: Create Repository on GitHub

### Option A: Via GitHub Website (Recommended)

1. **Go to GitHub**: https://github.com

2. **Create New Repository**:
   - Click the `+` icon (top right) → "New repository"
   - Or go to: https://github.com/new

3. **Repository Settings**:
   ```
   Repository name: fates-emulator
   Description: AutoML framework for FATES parameter calibration with coexistence constraints
   
   Visibility: 
     ○ Public (recommended - for community)
     ○ Private (if you want to review first)
   
   DO NOT initialize with:
     ☐ README (you already have one)
     ☐ .gitignore (you already have one)
     ☐ License (you already have one)
   ```

4. **Click "Create repository"**

### Option B: Via GitHub CLI (if installed)

```bash
# Install GitHub CLI first if needed
# https://cli.github.com/

# Create repository
gh repo create fates-emulator --public --source=. --remote=origin

# Or for private
gh repo create fates-emulator --private --source=. --remote=origin
```

---

## Step 3: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Use either method:

### Method A: SSH (Recommended for HPC)

```bash
# Add remote
git remote add origin git@github.com:YOUR_USERNAME/fates-emulator.git

# Verify
git remote -v

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### Method B: HTTPS

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/fates-emulator.git

# Set main branch
git branch -M main

# Push to GitHub (will ask for credentials)
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

---

## Step 4: Verify Upload

1. **Go to your repository**: https://github.com/YOUR_USERNAME/fates-emulator

2. **Check that you see**:
   - ✅ README.md displayed on main page
   - ✅ Badges showing (Python, License, DOI)
   - ✅ All folders (src/, docs/, workflows/, examples/)
   - ✅ 33 files committed

3. **Test links**:
   - Click on docs/00_overview.md
   - Check if examples/ has the config
   - Verify LICENSE is visible

---

## Step 5: Configure Repository Settings

### 5.1: Add Topics (for discoverability)

1. Go to your repo main page
2. Click the gear icon ⚙️ next to "About"
3. Add topics:
   ```
   fates
   machine-learning
   automl
   ecosystem-modeling
   parameter-calibration
   xgboost
   flaml
   shap
   python
   climate-modeling
   ```

### 5.2: Set Repository Description

In "About" section:
```
Description: AutoML framework for FATES parameter calibration with coexistence constraints

Website: https://doi.org/10.5194/gmd-16-4017-2023
```

### 5.3: Enable Features

Settings → General → Features:
- ✅ Issues (for bug reports)
- ✅ Projects (optional)
- ✅ Wiki (optional)
- ✅ Discussions (recommended for Q&A)

---

## Step 6: Create Release (Optional but Recommended)

1. **Go to Releases**: Click "Releases" on right side → "Create a new release"

2. **Tag version**: `v0.1.0`

3. **Release title**: `FATES-Emulator v0.1.0 - Initial Release`

4. **Description**: (Copy from CHANGELOG.md)
   ```markdown
   # FATES-Emulator v0.1.0
   
   Initial release of FATES-Emulator framework for ecosystem model calibration.
   
   ## Features
   - FLAML AutoML for automatic hyperparameter optimization
   - Parameter sampling with ecological constraints
   - SHAP interpretability analysis
   - Multi-objective calibration with coexistence
   - Complete documentation and examples
   - No hardcoded paths - works on any system
   
   ## Based on
   Li, L., et al. (2023). A machine learning approach targeting parameter 
   estimation for plant functional type coexistence modeling using ELM-FATES 
   (v2.0). Geosci. Model Dev., 16, 4017–4040.
   https://doi.org/10.5194/gmd-16-4017-2023
   
   ## Installation
   ```bash
   git clone https://github.com/YOUR_USERNAME/fates-emulator.git
   cd fates-emulator
   conda env create -f environment.yml
   conda activate fates-emulator
   pip install -e .
   ```
   
   See QUICKSTART.md for getting started.
   ```

5. **Click "Publish release"**

---

## Step 7: Get DOI from Zenodo (Optional)

To make your code citable:

1. **Link GitHub to Zenodo**:
   - Go to https://zenodo.org/
   - Log in (use GitHub account)
   - Go to https://zenodo.org/account/settings/github/
   - Find your repo and flip the switch ON

2. **Create Release** (if not done in Step 6)
   - Zenodo automatically archives each release

3. **Get DOI**:
   - Go to Zenodo, find your repository
   - Copy the DOI badge markdown
   - Add to your README.md

4. **Update README**:
   ```bash
   # Edit README locally
   nano README.md
   # Add Zenodo badge after other badges
   
   # Commit and push
   git add README.md
   git commit -m "Add Zenodo DOI badge"
   git push
   ```

---

## Step 8: Share with Community

### Announce on:

1. **FATES Forum**:
   - https://github.com/NGEET/fates/discussions
   - Post about your new tool

2. **E3SM Mailing List**:
   - Share with E3SM community

3. **Social Media** (optional):
   - Twitter/X: @mention relevant accounts
   - LinkedIn: Share with colleagues

### Example Post:
```
🎉 Introducing FATES-Emulator: An AutoML framework for FATES 
parameter calibration

Key features:
✅ FLAML AutoML (no manual tuning!)
✅ 100-1000× faster calibration
✅ SHAP interpretability
✅ Coexistence constraints
✅ Works on any HPC system

Based on our GMD 2023 paper: https://doi.org/10.5194/gmd-16-4017-2023

GitHub: https://github.com/YOUR_USERNAME/fates-emulator

#FATES #MachineLearning #ClimateModeling #OpenScience
```

---

## Troubleshooting

### Problem: "Permission denied (publickey)"

**Solution**: Set up SSH keys
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

### Problem: "Failed to push"

**Solution**: Pull first if repo has changes
```bash
git pull origin main --rebase
git push origin main
```

### Problem: "Large files rejected"

**Solution**: Check .gitignore caught them
```bash
# Remove large files from git
git rm --cached path/to/large/file
echo "path/to/large/file" >> .gitignore
git commit -m "Remove large files"
git push
```

---

## Quick Command Reference

```bash
# Status
git status                    # Check what changed
git log --oneline            # View commit history

# Add files
git add .                    # Add all changes
git add specific_file.py     # Add specific file

# Commit
git commit -m "message"      # Commit with message
git commit --amend          # Modify last commit

# Push/Pull
git push                     # Push to GitHub
git pull                     # Pull from GitHub

# Branches (for future development)
git branch feature-name      # Create branch
git checkout feature-name    # Switch to branch
git merge feature-name       # Merge branch
```

---

## Next Steps After Upload

1. ✅ **Test Clone**: Clone your repo on a different machine to verify
   ```bash
   git clone https://github.com/YOUR_USERNAME/fates-emulator.git
   cd fates-emulator
   conda env create -f environment.yml
   ```

2. ✅ **Add GitHub Actions** (CI/CD - optional):
   - Create `.github/workflows/tests.yml`
   - Run tests automatically on push

3. ✅ **Add Documentation Site** (optional):
   - Use GitHub Pages
   - Host docs/ as website

4. ✅ **Monitor Issues**:
   - Respond to community questions
   - Fix bugs reported by users

5. ✅ **Accept Contributions**:
   - Review pull requests
   - Merge community improvements

---

## Repository URLs

After upload, your repository will be at:

- **Main Page**: `https://github.com/YOUR_USERNAME/fates-emulator`
- **Clone URL**: `git@github.com:YOUR_USERNAME/fates-emulator.git`
- **Issues**: `https://github.com/YOUR_USERNAME/fates-emulator/issues`
- **Releases**: `https://github.com/YOUR_USERNAME/fates-emulator/releases`

---

## Making Repository More Discoverable

1. **Add to awesome lists**:
   - Awesome Climate Science
   - Awesome Earth System Modeling

2. **Submit to directories**:
   - Papers With Code
   - Research Software Directory

3. **Write Blog Post**:
   - Your institution's blog
   - Medium/Dev.to

4. **Present at Conferences**:
   - E3SM User Meeting
   - FATES Workshop
   - AGU/EGU posters

---

## Questions?

- **GitHub Help**: https://docs.github.com/
- **Git Tutorial**: https://git-scm.com/docs/gittutorial
- **Contact**: lingcheng.li@pnnl.gov

---

## ✅ Checklist Before Upload

- [ ] All files committed locally
- [ ] .gitignore excludes large files
- [ ] README looks good
- [ ] LICENSE present
- [ ] No sensitive data (passwords, keys)
- [ ] No absolute paths in code
- [ ] Documentation is clear
- [ ] Examples work

---

**Ready to share your work with the world!** 🚀

Good luck with your GitHub upload!

