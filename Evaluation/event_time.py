import json
import re
from datetime import datetime

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, levene

import pprint

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

TIME_RE = re.compile(r'"time"\s*:\s*"([^"]+)"')

def extract_nextcloud_timestamps(lines):
    timestamps = []
    for line in lines:
        match = TIME_RE.search(line)
        if match:
            timestamps.append(datetime.fromisoformat(match.group(1)))
    return timestamps


AUDIT_TIMESTAMP_RE = re.compile(r"audit\((\d+\.\d+):")

def extract_auditlog_timestamps(lines):
    timestamps = []
    for line in lines:
        match = AUDIT_TIMESTAMP_RE.search(line)
        if match:
            epoch_str = match.group(1)   # e.g. "1763731447.117"
            try:
                ts = datetime.fromtimestamp(float(epoch_str))
                timestamps.append(ts)
            except Exception:
                continue
    return timestamps


def extract_syslog_timestamps(lines):
    timestamps = []
    for line in lines:
        try:
            ts_str = line.split(" ")[0]   # everything before first space
            ts = datetime.fromisoformat(ts_str)
            timestamps.append(ts)
        except Exception:
            # skip malformed or unrelated lines
            continue
    return timestamps


def timestamp_diffs_seconds(timestamps):
    """
    Given a list of datetime objects, return a list of differences in seconds
    between each consecutive timestamp.
    """
    diffs = []
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        diffs.append(delta.total_seconds())
    return diffs


def stats_and_tests(list_a, list_b):
    """
    Compare two lists of numerical values (e.g., timestamp differences).
    Returns descriptive statistics + distribution comparison tests.
    """
    
    def describe(xs):
        xs = np.array(xs, dtype=float)
        if len(xs) == 0:
            return {"error": "empty list"}
        
        return {
            "count": len(xs),
            "mean": float(xs.mean()),
            "median": float(np.median(xs)),
            "std": float(xs.std()),
            "variance": float(xs.var()),
            "min": float(xs.min()),
            "max": float(xs.max()),
            "p5": float(np.percentile(xs, 5)),
            "p25": float(np.percentile(xs, 25)),
            "p75": float(np.percentile(xs, 75)),
            "p95": float(np.percentile(xs, 95)),
        }

    # Prepare outputs
    output = {
        "stats_a": describe(list_a),
        "stats_b": describe(list_b),
        "tests": {}
    }

    # Only run tests if both lists are non-empty
    if len(list_a) > 0 and len(list_b) > 0:
        a = np.array(list_a, dtype=float)
        b = np.array(list_b, dtype=float)

        # Kolmogorov–Smirnov test
        output["tests"]["ks_test"] = {
            "statistic": float(ks_2samp(a, b).statistic),
            "p_value": float(ks_2samp(a, b).pvalue),
        }

        # Mann-Whitney U test
        mw = mannwhitneyu(a, b, alternative="two-sided")
        output["tests"]["mann_whitney"] = {
            "statistic": float(mw.statistic),
            "p_value": float(mw.pvalue),
        }

        # Levene test (variance comparison)
        lev = levene(a, b)
        output["tests"]["levene"] = {
            "statistic": float(lev.statistic),
            "p_value": float(lev.pvalue),
        }

    return output



def plot_timestamp_distributions(list_a, list_b, labels=("A", "B")):
    """
    Create histogram, KDE, and ECDF plots for comparing two distributions.
    
    list_a, list_b: numeric timestamp-difference lists
    labels: tuple giving the names for list A and B
    """
    a = np.array(list_a, dtype=float)
    b = np.array(list_b, dtype=float)
    
    # Remove non-finite values (NaN, inf) just in case
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    label_a, label_b = labels

    # ---------------------------
    # 1) HISTOGRAMS (side by side)
    # ---------------------------
    plt.figure(figsize=(10, 5))
    bins = max(10, int(np.sqrt(len(a) + len(b))))  # rule of thumb

    plt.hist(a, bins=bins, alpha=0.5, label=f"{label_a} histogram")
    plt.hist(b, bins=bins, alpha=0.5, label=f"{label_b} histogram")

    plt.title("Histogram Comparison")
    plt.xlabel("Timestamp difference (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 2) KERNEL DENSITY ESTIMATE
    # ---------------------------
    plt.figure(figsize=(10, 5))

    # KDE: only valid if we have >1 data point
    if len(a) > 1:
        kde_a = gaussian_kde(a)
        xs = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 500)
        plt.plot(xs, kde_a(xs), label=f"{label_a} KDE")

    if len(b) > 1:
        kde_b = gaussian_kde(b)
        xs = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 500)
        plt.plot(xs, kde_b(xs), label=f"{label_b} KDE")

    plt.title("Kernel Density Estimate (KDE) Comparison")
    plt.xlabel("Timestamp difference (s)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 3) ECDF PLOTS
    # ---------------------------
    def ecdf(x):
        """Return sorted x and ECDF y-values."""
        x = np.sort(x)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    plt.figure(figsize=(10, 5))

    x_a, y_a = ecdf(a)
    x_b, y_b = ecdf(b)

    plt.plot(x_a, y_a, label=f"{label_a} ECDF", drawstyle='steps-post')
    plt.plot(x_b, y_b, label=f"{label_b} ECDF", drawstyle='steps-post')

    plt.title("ECDF Comparison")
    plt.xlabel("Timestamp difference (s)")
    plt.ylabel("Cumulative probability")
    plt.legend()
    plt.grid(True)
    plt.show()




syslog_lines = [
    "2025-11-21T13:24:32.510196+00:00 server systemd[1]: Reloading apache2.service...",
    "2025-11-21T13:24:32.694518+00:00 server systemd[1]: Reloaded apache2.service...",
    "2025-11-21T13:24:33.577005+00:00 server systemd[1]: Stopping apache2.service...",
]

timestamps = extract_syslog_timestamps(syslog_lines)

audit_lines = [
    'type=SYSCALL msg=audit(1763731447.117:55962): arch=c000003e syscall=59 ...',
    'type=EXECVE msg=audit(1763731447.117:55962): argc=4 ...',
    'type=SYSCALL msg=audit(1763731447.176:55963): arch=c000003e syscall=59 ...',
]

timestamps = extract_auditlog_timestamps(audit_lines)

nextcloud_lines = [
    '{"reqId":"RfZfVd1HITMQJg9HUUhc","level":0,"time":"2025-11-21T13:26:44+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"no app in context","method":"PROPFIND","url":"/remote.php/dav/files/Admin/","message":"The loading of lazy UserConfig values have been requested","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","exception":{"Exception":"RuntimeException","Message":"ignorable exception","Code":0,"Trace":[{"file":"/var/www/nextcloud/lib/private/Config/UserConfig.php","line":1685,"function":"loadConfig","class":"OC\\Config\\UserConfig","type":"->","args":["Admin",null]},{"file":"/var/www/nextcloud/lib/private/Config/UserConfig.php","line":132,"function":"loadConfigAll","class":"OC\\Config\\UserConfig","type":"->","args":["Admin"]},{"file":"/var/www/nextcloud/lib/private/AllConfig.php","line":298,"function":"getKeys","class":"OC\\Config\\UserConfig","type":"->","args":["Admin","login_token_2fa"]},{"file":"/var/www/nextcloud/lib/private/Authentication/TwoFactorAuth/Manager.php","line":324,"function":"getUserKeys","class":"OC\\AllConfig","type":"->","args":["Admin","login_token_2fa"]},{"file":"/var/www/nextcloud/apps/dav/lib/Connector/Sabre/Auth.php","line":174,"function":"needsSecondFactor","class":"OC\\Authentication\\TwoFactorAuth\\Manager","type":"->","args":[{"__class__":"OC\\User\\User"}]},{"file":"/var/www/nextcloud/apps/dav/lib/Connector/Sabre/Auth.php","line":105,"function":"auth","class":"OCA\\DAV\\Connector\\Sabre\\Auth","type":"->","args":[{"__class__":"Sabre\\HTTP\\Request"},{"__class__":"Sabre\\HTTP\\Response"}]},{"file":"/var/www/nextcloud/3rdparty/sabre/dav/lib/DAV/Auth/Plugin.php","line":179,"function":"check","class":"OCA\\DAV\\Connector\\Sabre\\Auth","type":"->","args":[{"__class__":"Sabre\\HTTP\\Request"},{"__class__":"Sabre\\HTTP\\Response"}]},{"file":"/var/www/nextcloud/3rdparty/sabre/dav/lib/DAV/Auth/Plugin.php","line":135,"function":"check","class":"Sabre\\DAV\\Auth\\Plugin","type":"->","args":[{"__class__":"Sabre\\HTTP\\Request"},{"__class__":"Sabre\\HTTP\\Response"}]},{"file":"/var/www/nextcloud/3rdparty/sabre/event/lib/WildcardEmitterTrait.php","line":89,"function":"beforeMethod","class":"Sabre\\DAV\\Auth\\Plugin","type":"->","args":[{"__class__":"Sabre\\HTTP\\Request"},{"__class__":"Sabre\\HTTP\\Response"}]},{"file":"/var/www/nextcloud/3rdparty/sabre/dav/lib/DAV/Server.php","line":456,"function":"emit","class":"Sabre\\DAV\\Server","type":"->","args":["beforeMethod:PROPFIND",[{"__class__":"Sabre\\HTTP\\Request"},{"__class__":"Sabre\\HTTP\\Response"}]]},{"file":"/var/www/nextcloud/apps/dav/lib/Connector/Sabre/Server.php","line":49,"function":"invokeMethod","class":"Sabre\\DAV\\Server","type":"->","args":[{"__class__":"Sabre\\HTTP\\Request"},{"__class__":"Sabre\\HTTP\\Response"}]},{"file":"/var/www/nextcloud/apps/dav/lib/Server.php","line":401,"function":"start","class":"OCA\\DAV\\Connector\\Sabre\\Server","type":"->","args":[]},{"file":"/var/www/nextcloud/apps/dav/appinfo/v2/remote.php","line":21,"function":"exec","class":"OCA\\DAV\\Server","type":"->","args":[]},{"file":"/var/www/nextcloud/remote.php","line":145,"args":["/var/www/nextcloud/apps/dav/appinfo/v2/remote.php"],"function":"require_once"}],"File":"/var/www/nextcloud/lib/private/Config/UserConfig.php","Line":1699,"message":"The loading of lazy UserConfig values have been requested","exception":{},"CustomMessage":"The loading of lazy UserConfig values have been requested"}}',
    '{"reqId":"lgaxC4Eo86uwyDQc4K2X","level":0,"time":"2025-11-21T13:26:44+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"no app in context","method":"GET","url":"/ocs/v2.php/apps/recommendations/api/v1/recommendations/always","message":"OCA\\Recommendations\\Controller\\RecommendationController::always uses the @NoAdminRequired annotation and should use the #[OCP\\AppFramework\\Http\\Attribute\\NoAdminRequired] attribute instead","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":[]}',
    '{"reqId":"lgaxC4Eo86uwyDQc4K2X","level":0,"time":"2025-11-21T13:26:44+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"no app in context","method":"GET","url":"/ocs/v2.php/apps/recommendations/api/v1/recommendations/always","message":"OCA\\Recommendations\\Controller\\RecommendationController::always uses the @NoAdminRequired annotation and should use the #[OCP\\AppFramework\\Http\\Attribute\\NoAdminRequired] attribute instead","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":[]}',
    '{"reqId":"2tClVap5iotMw91Byeqg","level":0,"time":"2025-11-21T13:26:45+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"no app in context","method":"GET","url":"/ocs/v2.php/apps/user_status/api/v1/user_status","message":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon.","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","exception":{"Exception":"Exception","Message":"No parameters in call to OC\\DB\\QueryBuilder\\ExpressionBuilder\\ExpressionBuilder::orX","Code":0,"Trace":[{"file":"/var/www/nextcloud/apps/circles/lib/Db/CircleRequest.php","line":268,"function":"orX","class":"OC\\DB\\QueryBuilder\\ExpressionBuilder\\ExpressionBuilder","type":"->","args":[]},{"file":"/var/www/nextcloud/apps/circles/lib/Db/CircleRequest.php","line":231,"function":"buildProbeCircle","class":"OCA\\Circles\\Db\\CircleRequest","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\FederatedUser"},{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"},{"__class__":"OCA\\Circles\\Model\\Probes\\DataProbe"}]},{"file":"/var/www/nextcloud/apps/circles/lib/Service/CircleService.php","line":808,"function":"probeCircles","class":"OCA\\Circles\\Db\\CircleRequest","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\FederatedUser"},{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"},{"__class__":"OCA\\Circles\\Model\\Probes\\DataProbe"}]},{"file":"/var/www/nextcloud/apps/circles/lib/Api/v1/Circles.php","line":134,"function":"probeCircles","class":"OCA\\Circles\\Service\\CircleService","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"}]},{"file":"/var/www/nextcloud/apps/dav/lib/Connector/Sabre/Principal.php","line":546,"function":"joinedCircles","class":"OCA\\Circles\\Api\\v1\\Circles","type":"::","args":["Admin",true]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":371,"function":"getCircleMembership","class":"OCA\\DAV\\Connector\\Sabre\\Principal","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/lib/public/AppFramework/Db/TTransactional.php","line":45,"function":"OCA\\DAV\\CalDAV\\{closure}","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":315,"function":"atomic","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":[{"__class__":"Closure"},{"__class__":"OC\\DB\\ConnectionAdapter"}]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalendarProvider.php","line":31,"function":"getCalendarsForUser","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":188,"function":"getCalendars","class":"OCA\\DAV\\CalDAV\\CalendarProvider","type":"->","args":["principals/users/Admin",[]]},{"function":"OC\\Calendar\\{closure}","class":"OC\\Calendar\\Manager","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":177,"function":"array_map","args":[{"__class__":"Closure"},["*** sensitive parameters replaced ***","*** sensitive parameters replaced ***"]]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/Status/StatusService.php","line":152,"function":"getCalendarsForPrincipal","class":"OC\\Calendar\\Manager","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/Status/StatusService.php","line":56,"function":"getCalendarEvents","class":"OCA\\DAV\\CalDAV\\Status\\StatusService","type":"->","args":[{"__class__":"OC\\User\\User"}]},{"file":"/var/www/nextcloud/apps/user_status/lib/Controller/UserStatusController.php","line":59,"function":"processCalendarStatus","class":"OCA\\DAV\\CalDAV\\Status\\StatusService","type":"->","args":["Admin"]},{"file":"/var/www/nextcloud/lib/private/AppFramework/Http/Dispatcher.php","line":200,"function":"getStatus","class":"OCA\\UserStatus\\Controller\\UserStatusController","type":"->","args":[]},{"file":"/var/www/nextcloud/lib/private/AppFramework/Http/Dispatcher.php","line":114,"function":"executeController","class":"OC\\AppFramework\\Http\\Dispatcher","type":"->","args":[{"__class__":"OCA\\UserStatus\\Controller\\UserStatusController"},"getStatus"]},{"file":"/var/www/nextcloud/lib/private/AppFramework/App.php","line":161,"function":"dispatch","class":"OC\\AppFramework\\Http\\Dispatcher","type":"->","args":[{"__class__":"OCA\\UserStatus\\Controller\\UserStatusController"},"getStatus"]},{"file":"/var/www/nextcloud/lib/private/Route/Router.php","line":315,"function":"main","class":"OC\\AppFramework\\App","type":"::","args":["OCA\\UserStatus\\Controller\\UserStatusController","getStatus",{"__class__":"OC\\AppFramework\\DependencyInjection\\DIContainer"},{"_route":"ocs.user_status.userstatus.getstatus"}]},{"file":"/var/www/nextcloud/ocs/v1.php","line":49,"function":"match","class":"OC\\Route\\Router","type":"->","args":["/ocsapp/apps/user_status/api/v1/user_status"]},{"file":"/var/www/nextcloud/ocs/v2.php","line":7,"args":["/var/www/nextcloud/ocs/v1.php"],"function":"require_once"}],"File":"/var/www/nextcloud/lib/private/DB/QueryBuilder/ExpressionBuilder/ExpressionBuilder.php","Line":87,"message":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon.","exception":{},"CustomMessage":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon."}}',
    '{"reqId":"2tClVap5iotMw91Byeqg","level":0,"time":"2025-11-21T13:26:45+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"no app in context","method":"GET","url":"/ocs/v2.php/apps/user_status/api/v1/user_status","message":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon.","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","exception":{"Exception":"Exception","Message":"No parameters in call to OC\\DB\\QueryBuilder\\ExpressionBuilder\\ExpressionBuilder::orX","Code":0,"Trace":[{"file":"/var/www/nextcloud/apps/circles/lib/Db/CircleRequest.php","line":268,"function":"orX","class":"OC\\DB\\QueryBuilder\\ExpressionBuilder\\ExpressionBuilder","type":"->","args":[]},{"file":"/var/www/nextcloud/apps/circles/lib/Db/CircleRequest.php","line":231,"function":"buildProbeCircle","class":"OCA\\Circles\\Db\\CircleRequest","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\FederatedUser"},{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"},{"__class__":"OCA\\Circles\\Model\\Probes\\DataProbe"}]},{"file":"/var/www/nextcloud/apps/circles/lib/Service/CircleService.php","line":808,"function":"probeCircles","class":"OCA\\Circles\\Db\\CircleRequest","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\FederatedUser"},{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"},{"__class__":"OCA\\Circles\\Model\\Probes\\DataProbe"}]},{"file":"/var/www/nextcloud/apps/circles/lib/Api/v1/Circles.php","line":134,"function":"probeCircles","class":"OCA\\Circles\\Service\\CircleService","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"}]},{"file":"/var/www/nextcloud/apps/dav/lib/Connector/Sabre/Principal.php","line":546,"function":"joinedCircles","class":"OCA\\Circles\\Api\\v1\\Circles","type":"::","args":["Admin",true]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":371,"function":"getCircleMembership","class":"OCA\\DAV\\Connector\\Sabre\\Principal","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/lib/public/AppFramework/Db/TTransactional.php","line":45,"function":"OCA\\DAV\\CalDAV\\{closure}","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":315,"function":"atomic","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":[{"__class__":"Closure"},{"__class__":"OC\\DB\\ConnectionAdapter"}]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalendarProvider.php","line":31,"function":"getCalendarsForUser","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":188,"function":"getCalendars","class":"OCA\\DAV\\CalDAV\\CalendarProvider","type":"->","args":["principals/users/Admin",["personal"]]},{"function":"OC\\Calendar\\{closure}","class":"OC\\Calendar\\Manager","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":177,"function":"array_map","args":[{"__class__":"Closure"},["*** sensitive parameters replaced ***","*** sensitive parameters replaced ***"]]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":195,"function":"getCalendarsForPrincipal","class":"OC\\Calendar\\Manager","type":"->","args":["principals/users/Admin",["personal"]]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/Status/StatusService.php","line":181,"function":"searchForPrincipal","class":"OC\\Calendar\\Manager","type":"->","args":[{"__class__":"OC\\Calendar\\CalendarQuery","searchProperties":[]}]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/Status/StatusService.php","line":56,"function":"getCalendarEvents","class":"OCA\\DAV\\CalDAV\\Status\\StatusService","type":"->","args":[{"__class__":"OC\\User\\User"}]},{"file":"/var/www/nextcloud/apps/user_status/lib/Controller/UserStatusController.php","line":59,"function":"processCalendarStatus","class":"OCA\\DAV\\CalDAV\\Status\\StatusService","type":"->","args":["Admin"]},{"file":"/var/www/nextcloud/lib/private/AppFramework/Http/Dispatcher.php","line":200,"function":"getStatus","class":"OCA\\UserStatus\\Controller\\UserStatusController","type":"->","args":[]},{"file":"/var/www/nextcloud/lib/private/AppFramework/Http/Dispatcher.php","line":114,"function":"executeController","class":"OC\\AppFramework\\Http\\Dispatcher","type":"->","args":[{"__class__":"OCA\\UserStatus\\Controller\\UserStatusController"},"getStatus"]},{"file":"/var/www/nextcloud/lib/private/AppFramework/App.php","line":161,"function":"dispatch","class":"OC\\AppFramework\\Http\\Dispatcher","type":"->","args":[{"__class__":"OCA\\UserStatus\\Controller\\UserStatusController"},"getStatus"]},{"file":"/var/www/nextcloud/lib/private/Route/Router.php","line":315,"function":"main","class":"OC\\AppFramework\\App","type":"::","args":["OCA\\UserStatus\\Controller\\UserStatusController","getStatus",{"__class__":"OC\\AppFramework\\DependencyInjection\\DIContainer"},{"_route":"ocs.user_status.userstatus.getstatus"}]},{"file":"/var/www/nextcloud/ocs/v1.php","line":49,"function":"match","class":"OC\\Route\\Router","type":"->","args":["/ocsapp/apps/user_status/api/v1/user_status"]},{"file":"/var/www/nextcloud/ocs/v2.php","line":7,"args":["/var/www/nextcloud/ocs/v1.php"],"function":"require_once"}],"File":"/var/www/nextcloud/lib/private/DB/QueryBuilder/ExpressionBuilder/ExpressionBuilder.php","Line":87,"message":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon.","exception":{},"CustomMessage":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon."}}',
    '{"reqId":"2tClVap5iotMw91Byeqg","level":0,"time":"2025-11-21T13:26:45+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"dav","method":"GET","url":"/ocs/v2.php/apps/user_status/api/v1/user_status","message":"No calendar events found for status check","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":{"app":"dav","user":"Admin"}}',
    '{"reqId":"I3Ajs2CvAmkoBCLrh50x","level":0,"time":"2025-11-21T13:26:46+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"dav","method":"PUT","url":"/ocs/v2.php/apps/user_status/api/v1/heartbeat?format=json","message":"No calendar events found for status check","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":{"app":"dav","user":"Admin"}}',
    '{"reqId":"tgevsbANkOVbuYpc6DY7","level":0,"time":"2025-11-21T13:26:46+00:00","remoteAddr":"192.168.56.106","user":"--","app":"cron","method":"GET","url":"/cron.php","message":"WebCron call has selected job with ID 155","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":{"app":"cron"}}',
    '{"reqId":"tgevsbANkOVbuYpc6DY7","level":0,"time":"2025-11-21T13:26:46+00:00","remoteAddr":"192.168.56.106","user":"--","app":"cron","method":"GET","url":"/cron.php","message":"Starting job OCA\\DAV\\BackgroundJob\\GenerateBirthdayCalendarBackgroundJob (id: 155, arguments: {\"userId\":\"Admin\",\"purgeBeforeGenerating\":true})","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":{"app":"cron"}}',
    '{"reqId":"tgevsbANkOVbuYpc6DY7","level":0,"time":"2025-11-21T13:26:46+00:00","remoteAddr":"192.168.56.106","user":"--","app":"dav","method":"GET","url":"/cron.php","message":"Activity generated for new calendar 3","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":{"app":"dav"}}',
    '{"reqId":"tgevsbANkOVbuYpc6DY7","level":0,"time":"2025-11-21T13:26:46+00:00","remoteAddr":"192.168.56.106","user":"--","app":"no app in context","method":"GET","url":"/cron.php","message":"dirty table reads: SELECT `displayname`, `description`, `timezone`, `calendarorder`, `calendarcolor`, `deleted_at`, `id`, `uri`, `synctoken`, `components`, `principaluri`, `transparent` FROM `*PREFIX*calendars` WHERE (`uri` = :dcValue1) AND (`principaluri` = :dcValue2) LIMIT 1","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","exception":{"Exception":"Exception","Message":"dirty table reads: SELECT `displayname`, `description`, `timezone`, `calendarorder`, `calendarcolor`, `deleted_at`, `id`, `uri`, `synctoken`, `components`, `principaluri`, `transparent` FROM `*PREFIX*calendars` WHERE (`uri` = :dcValue1) AND (`principaluri` = :dcValue2) LIMIT 1","Code":0,"Trace":[{"file":"/var/www/nextcloud/lib/private/DB/ConnectionAdapter.php","line":50,"function":"executeQuery","class":"OC\\DB\\Connection","type":"->","args":["SELECT `displayname`, `description`, `timezone`, `calendarorder`, `calendarcolor`, `deleted_at`, `id`, `uri`, `synctoken`, `components`, `principaluri`, `transparent` FROM `*PREFIX*calendars` WHERE (`uri` = :dcValue1) AND (`principaluri` = :dcValue2) LIMIT 1",{"dcValue1":"contact_birthdays","dcValue2":"principals/users/Admin"},{"dcValue1":2,"dcValue2":2}]},{"file":"/var/www/nextcloud/lib/private/DB/QueryBuilder/QueryBuilder.php","line":289,"function":"executeQuery","class":"OC\\DB\\ConnectionAdapter","type":"->","args":["SELECT `displayname`, `description`, `timezone`, `calendarorder`, `calendarcolor`, `deleted_at`, `id`, `uri`, `synctoken`, `components`, `principaluri`, `transparent` FROM `*PREFIX*calendars` WHERE (`uri` = :dcValue1) AND (`principaluri` = :dcValue2) LIMIT 1",{"dcValue1":"contact_birthdays","dcValue2":"principals/users/Admin"},{"dcValue1":2,"dcValue2":2}]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":637,"function":"executeQuery","class":"OC\\DB\\QueryBuilder\\QueryBuilder","type":"->","args":[]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/BirthdayService.php","line":120,"function":"getCalendarByUri","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["principals/users/Admin","contact_birthdays"]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/BirthdayService.php","line":274,"function":"ensureCalendarExists","class":"OCA\\DAV\\CalDAV\\BirthdayService","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/apps/dav/lib/BackgroundJob/GenerateBirthdayCalendarBackgroundJob.php","line":46,"function":"syncUser","class":"OCA\\DAV\\CalDAV\\BirthdayService","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/lib/public/BackgroundJob/Job.php","line":61,"function":"run","class":"OCA\\DAV\\BackgroundJob\\GenerateBirthdayCalendarBackgroundJob","type":"->","args":[{"userId":"*** sensitive parameters replaced ***","purgeBeforeGenerating":true}]},{"file":"/var/www/nextcloud/lib/public/BackgroundJob/QueuedJob.php","line":43,"function":"start","class":"OCP\\BackgroundJob\\Job","type":"->","args":[{"__class__":"OC\\BackgroundJob\\JobList"}]},{"file":"/var/www/nextcloud/lib/public/BackgroundJob/QueuedJob.php","line":29,"function":"start","class":"OCP\\BackgroundJob\\QueuedJob","type":"->","args":[{"__class__":"OC\\BackgroundJob\\JobList"}]},{"file":"/var/www/nextcloud/cron.php","line":236,"function":"execute","class":"OCP\\BackgroundJob\\QueuedJob","type":"->","args":[{"__class__":"OC\\BackgroundJob\\JobList"}]}],"File":"/var/www/nextcloud/lib/private/DB/Connection.php","Line":406,"message":"dirty table reads: SELECT `displayname`, `description`, `timezone`, `calendarorder`, `calendarcolor`, `deleted_at`, `id`, `uri`, `synctoken`, `components`, `principaluri`, `transparent` FROM `*PREFIX*calendars` WHERE (`uri` = :dcValue1) AND (`principaluri` = :dcValue2) LIMIT 1","tables":["oc_jobs","oc_calendars","oc_activity"],"reads":["oc_calendars"],"exception":{},"CustomMessage":"dirty table reads: SELECT `displayname`, `description`, `timezone`, `calendarorder`, `calendarcolor`, `deleted_at`, `id`, `uri`, `synctoken`, `components`, `principaluri`, `transparent` FROM `*PREFIX*calendars` WHERE (`uri` = :dcValue1) AND (`principaluri` = :dcValue2) LIMIT 1"}}',
    '{"reqId":"tgevsbANkOVbuYpc6DY7","level":0,"time":"2025-11-21T13:26:46+00:00","remoteAddr":"192.168.56.106","user":"--","app":"cron","method":"GET","url":"/cron.php","message":"Finished job OCA\\DAV\\BackgroundJob\\GenerateBirthdayCalendarBackgroundJob (id: 155, arguments: {\"userId\":\"Admin\",\"purgeBeforeGenerating\":true}) in 0 seconds","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","data":{"app":"cron"}}',
    '{"reqId":"1zNYLWotI7JWEkDpepgX","level":0,"time":"2025-11-21T13:26:47+00:00","remoteAddr":"192.168.56.106","user":"Admin","app":"no app in context","method":"GET","url":"/ocs/v2.php/apps/dashboard/api/v2/widget-items?widgets%5B%5D=calendar","message":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon.","userAgent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36","version":"31.0.7.1","exception":{"Exception":"Exception","Message":"No parameters in call to OC\\DB\\QueryBuilder\\ExpressionBuilder\\ExpressionBuilder::orX","Code":0,"Trace":[{"file":"/var/www/nextcloud/apps/circles/lib/Db/CircleRequest.php","line":268,"function":"orX","class":"OC\\DB\\QueryBuilder\\ExpressionBuilder\\ExpressionBuilder","type":"->","args":[]},{"file":"/var/www/nextcloud/apps/circles/lib/Db/CircleRequest.php","line":231,"function":"buildProbeCircle","class":"OCA\\Circles\\Db\\CircleRequest","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\FederatedUser"},{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"},{"__class__":"OCA\\Circles\\Model\\Probes\\DataProbe"}]},{"file":"/var/www/nextcloud/apps/circles/lib/Service/CircleService.php","line":808,"function":"probeCircles","class":"OCA\\Circles\\Db\\CircleRequest","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\FederatedUser"},{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"},{"__class__":"OCA\\Circles\\Model\\Probes\\DataProbe"}]},{"file":"/var/www/nextcloud/apps/circles/lib/Api/v1/Circles.php","line":134,"function":"probeCircles","class":"OCA\\Circles\\Service\\CircleService","type":"->","args":[{"__class__":"OCA\\Circles\\Model\\Probes\\CircleProbe"}]},{"file":"/var/www/nextcloud/apps/dav/lib/Connector/Sabre/Principal.php","line":546,"function":"joinedCircles","class":"OCA\\Circles\\Api\\v1\\Circles","type":"::","args":["Admin",true]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":371,"function":"getCircleMembership","class":"OCA\\DAV\\Connector\\Sabre\\Principal","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/lib/public/AppFramework/Db/TTransactional.php","line":45,"function":"OCA\\DAV\\CalDAV\\{closure}","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalDavBackend.php","line":315,"function":"atomic","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":[{"__class__":"Closure"},{"__class__":"OC\\DB\\ConnectionAdapter"}]},{"file":"/var/www/nextcloud/apps/dav/lib/CalDAV/CalendarProvider.php","line":31,"function":"getCalendarsForUser","class":"OCA\\DAV\\CalDAV\\CalDavBackend","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":188,"function":"getCalendars","class":"OCA\\DAV\\CalDAV\\CalendarProvider","type":"->","args":["principals/users/Admin",[]]},{"function":"OC\\Calendar\\{closure}","class":"OC\\Calendar\\Manager","type":"->","args":["*** sensitive parameters replaced ***"]},{"file":"/var/www/nextcloud/lib/private/Calendar/Manager.php","line":177,"function":"array_map","args":[{"__class__":"Closure"},["*** sensitive parameters replaced ***","*** sensitive parameters replaced ***"]]},{"file":"/var/www/nextcloud/apps/calendar/lib/Dashboard/CalendarWidget.php","line":120,"function":"getCalendarsForPrincipal","class":"OC\\Calendar\\Manager","type":"->","args":["principals/users/Admin"]},{"file":"/var/www/nextcloud/apps/calendar/lib/Dashboard/CalendarWidget.php","line":181,"function":"getItems","class":"OCA\\Calendar\\Dashboard\\CalendarWidget","type":"->","args":["Admin",null,7]},{"file":"/var/www/nextcloud/apps/dashboard/lib/Controller/DashboardApiController.php","line":119,"function":"getItemsV2","class":"OCA\\Calendar\\Dashboard\\CalendarWidget","type":"->","args":["Admin",null,7]},{"file":"/var/www/nextcloud/lib/private/AppFramework/Http/Dispatcher.php","line":200,"function":"getWidgetItemsV2","class":"OCA\\Dashboard\\Controller\\DashboardApiController","type":"->","args":[[],7,{"calendar":{"__class__":"OCA\\Calendar\\Dashboard\\CalendarWidget"}}]},{"file":"/var/www/nextcloud/lib/private/AppFramework/Http/Dispatcher.php","line":114,"function":"executeController","class":"OC\\AppFramework\\Http\\Dispatcher","type":"->","args":[{"__class__":"OCA\\Dashboard\\Controller\\DashboardApiController"},"getWidgetItemsV2"]},{"file":"/var/www/nextcloud/lib/private/AppFramework/App.php","line":161,"function":"dispatch","class":"OC\\AppFramework\\Http\\Dispatcher","type":"->","args":[{"__class__":"OCA\\Dashboard\\Controller\\DashboardApiController"},"getWidgetItemsV2"]},{"file":"/var/www/nextcloud/lib/private/Route/Router.php","line":315,"function":"main","class":"OC\\AppFramework\\App","type":"::","args":["OCA\\Dashboard\\Controller\\DashboardApiController","getWidgetItemsV2",{"__class__":"OC\\AppFramework\\DependencyInjection\\DIContainer"},{"_route":"ocs.dashboard.dashboardapi.getwidgetitemsv2"}]},{"file":"/var/www/nextcloud/ocs/v1.php","line":49,"function":"match","class":"OC\\Route\\Router","type":"->","args":["/ocsapp/apps/dashboard/api/v2/widget-items"]},{"file":"/var/www/nextcloud/ocs/v2.php","line":7,"args":["/var/www/nextcloud/ocs/v1.php"],"function":"require_once"}],"File":"/var/www/nextcloud/lib/private/DB/QueryBuilder/ExpressionBuilder/ExpressionBuilder.php","Line":87,"message":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon.","exception":{},"CustomMessage":"Calling OCP\\DB\\QueryBuilder\\IQueryBuilder::orX without parameters is deprecated and will throw soon."}}',
]

timestamps = extract_nextcloud_timestamps(nextcloud_lines)



diffs1 = [0.5, 0.7, 0.9, 1.1, 2.0]
diffs2 = [1.0, 1.2, 1.4, 1.6, 3.0]



plot_timestamp_distributions(diffs1, diffs2, labels=("Nextcloud", "Syslog"))
